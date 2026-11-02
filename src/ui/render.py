"""Render saved numerical results; charts never refetch or synthesize prices."""
import pandas as pd
import streamlit as st
import re
from dataclasses import replace
from ui.sources import link_source_pdfs, source_pdf_url
from agents.chat.agent import DISCLAIMER


def response_text(text):
    """The shared page footer contains the disclaimer, once per page."""
    pattern = r'(?:\*{0,2}Disclaimer:?\*{0,2}\s*)?\*{0,2}' + re.escape(DISCLAIMER) + r'\*{0,2}'
    return re.sub(pattern, '', text, flags=re.I).strip()


def charts(agent, report, *, key):
    if agent == 'portfolio':
        analysis = report['analysis']
        columns = st.columns(3)
        columns[0].metric('Portfolio value · USD', f"${analysis['calculated_value_usd']:,.0f}")
        columns[1].metric('Largest position', f"{analysis['top_holding_pct']:.1f}%")
        columns[2].metric('Effective holdings', f"{analysis['effective_holdings']:.1f}")
        st.caption('Values and sectors come from your supplied holdings.')
        mix = pd.DataFrame([{'Asset class': k, 'Weight (%)': v} for k, v in analysis['asset_allocation_pct'].items()])
        st.bar_chart(mix, x='Asset class', y='Weight (%)', horizontal=True, color='#167D75')
        if analysis.get('sector_allocation_pct'):
            with st.expander('Sector exposure'):
                sectors = pd.DataFrame([{'Sector': k, 'Weight (%)': v} for k, v in analysis['sector_allocation_pct'].items()])
                st.bar_chart(sectors, x='Sector', y='Weight (%)', horizontal=True, color='#648B68')
    elif agent == 'market':
        metrics = report.get('metrics')
        if not metrics:
            st.info('Market analysis is unavailable right now. Please try again later.')
            return
        cols = st.columns(3)
        cols[0].metric('Period return', f"{metrics['period_return_pct']:.2f}%")
        vol = metrics['annualized_volatility_pct']
        cols[1].metric('Annualized volatility', f'{vol:.2f}%' if vol is not None else 'Unavailable')
        cols[2].metric('Maximum drawdown', f"{metrics['max_drawdown_pct']:.2f}%")
        st.caption(f"{report['symbol']} · {metrics['start_date']} to {metrics['end_date']} · {report['price_basis'].replace('_', ' ')} · end-of-day data")
        rows = report.get('observed_prices', [])
        if rows:
            st.line_chart(pd.DataFrame(rows), x='date', y='close', color='#167D75', x_label='Observed date', y_label='Closing price (USD)')
        if metrics.get('benchmark_return_pct') is not None:
            st.caption(f"{report['benchmark_symbol']} returned {metrics['benchmark_return_pct']:.2f}%; difference {metrics['benchmark_difference_pp']:.2f} percentage points.")
    elif agent == 'goals':
        currency = report['plan']['currency']
        for result in report['results']:
            projection = result['projection']
            st.markdown('#### ' + result['goal']['name'])
            columns = st.columns(3)
            columns[0].metric(f'Projected balance · {currency}', f"{projection['ending_balance']:,.0f}")
            columns[1].metric('Target at deadline', f"{projection['future_target']:,.0f}")
            columns[2].metric('Shortfall', f"{projection['shortfall']:,.0f}")
            rows = [{'date': projection['start_date'], 'Projected balance': result['goal']['starting_balance'],
                     'Target at deadline': projection['future_target']}]
            rows.extend({'date': row['date'], 'Projected balance': row['closing_balance'],
                         'Target at deadline': projection['future_target']} for row in projection['schedule'])
            st.line_chart(pd.DataFrame(rows), x='date', y=['Projected balance', 'Target at deadline'], color=['#167D75', '#C38842'])
        st.caption('Projections depend on your assumptions; they are not forecasts or probabilities of success.')


def _sources(report):
    sources = {}
    knowledge = report.get('knowledge') or []
    groups = [section.get('retrieval', {}) for section in knowledge] if isinstance(knowledge, list) else [knowledge]
    for group in groups:
        for chunk in group.get('chunks', []):
            sources.setdefault(chunk['source'], set()).add(chunk['page'])
    for reference in report.get('references', []):
        sources.setdefault(reference['source'], set()).add(reference['page'])
    return sources


def report_response(memory, response, *, key, show_charts=True):
    agent = response.get('agent')
    result = response.get('result')
    if response['status'] == 'needs_input':
        st.info('Please complete the required inputs to continue.')
        return
    if not result or not result.get('report_id'):
        st.info('We couldn’t prepare an answer right now. Please try again.')
        return
    service = memory.chat_service if agent == 'chat' else memory.service
    thread_id = memory.chat_thread_id if agent == 'chat' else memory.thread_id
    saved = service.get_report(thread_id, result['report_id'])
    report = saved['report']
    sources = _sources(report)
    markdown = link_source_pdfs(response_text(response['message']), sources)
    if result['status'] in {'partial', 'unavailable'}:
        st.warning('Some parts of the analysis are unavailable. Showing the available results.')
    if show_charts:
        charts(agent, report, key=key)
    native = service.store.native(thread_id, result['report_id'])
    explanation = getattr(native, 'explanation', None)
    if explanation:
        if getattr(explanation, 'fallback_reason', None):
            explanation = replace(explanation, fallback_reason=None, method='Calculated summary')
        st.markdown(response_text(explanation.to_markdown()))
        if response.get('metrics', {}).get('reused_report'):
            st.caption('This explanation uses the saved report. Data and assumptions were not refreshed.')
        # Analytical explanations cite numbered excerpts; show their links once
        # inline. Chat answers already include their own complete reference list.
        references = []
        citation = 0
        knowledge = report.get('knowledge') or []
        groups = [section.get('retrieval', {}) for section in knowledge] if isinstance(knowledge, list) else [knowledge]
        for group in groups:
            for chunk in group.get('chunks', []):
                citation += 1
                label = f"{chunk['source']}, p. {chunk['page']}"
                url = source_pdf_url(chunk['source'], chunk['page'])
                references.append(f'[{citation}] [{label}]({url})' if url else f'[{citation}] {label}')
        if references:
            st.markdown('**References**\n\n' + '\n\n'.join(references))
    else:
        st.markdown(markdown)
