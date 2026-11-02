"""Finnie Streamlit workspace. Execute only from explicit form/chat submissions."""
from __future__ import annotations

from datetime import date
import json

from dotenv import load_dotenv
from pydantic import ValidationError
import streamlit as st

from agents.portfolio.holdings import MAX_FILE_BYTES
from agents.portfolio.upload import TEMPLATE
from rag.ingest import PROJECT_ROOT
from ui.controller import SessionMemory
from ui.render import report_response
from tools.diagnostics import log_event


def _clear_widgets():
    for key in list(st.session_state):
        if key.startswith('work_'):
            del st.session_state[key]


def _refresh(memory):
    # Apply programmatic selections before widgets are instantiated next run.
    st.session_state['_next_inputs'] = ['work_portfolio_choice', 'work_plan_choice']
    st.rerun()


def _error(exc):
    if isinstance(exc, ValidationError):
        return '\n'.join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False, include_url=False))
    return str(exc)


def _action(function, *, success=None):
    try:
        result = function()
        if success:
            st.session_state['_notice'] = success
        return result
    except Exception as exc:
        log_event('ui_action_failed', error_type=type(exc).__name__)
        st.warning('We couldn’t complete that request. Please check your inputs and try again.')
        return None


def _run(memory, message, *, route=None, arguments=None, resume=None):
    with st.spinner('Finnie is preparing your analysis…'):
        response = _action(lambda: memory.ask(message, route=route, arguments=arguments, resume=resume,
                                             use_llm=True, use_rag=True))
    if response is not None:
        if route == 'goals' and arguments and arguments.get('plan_id'):
            selected = arguments['plan_id']
            memory.selected_plan = selected
            if not hasattr(memory, 'goal_results'):
                memory.goal_results = {}
            memory.goal_results[selected] = response.model_dump(mode='json')
        _refresh(memory)


def _sidebar(memory):
    with st.sidebar:
        st.markdown('## Finnie')
        st.caption('Your financial workspace')
        profiles = memory.profiles.list()
        names = {'guest': 'Guest · session only', **{p.id: p.name for p in profiles}}
        choice = st.selectbox('Local profile', list(names), format_func=names.get, key='profile_choice')
        if choice != memory.profile_key:
            memory.switch_profile(choice)
            _clear_widgets()
            st.rerun()
        with st.expander('Create a local profile'):
            with st.form('create_profile'):
                name = st.text_input('Profile name', max_chars=80)
                create = st.form_submit_button('Create profile')
            if create:
                identifier = _action(lambda: memory.create_profile(name))
                if identifier:
                    st.session_state['_next_profile'] = identifier
                    _clear_widgets()
                    st.rerun()
        st.caption('Profiles are local to this computer. This MVP has no account sign-in.')
        if st.button('New conversation', width='stretch'):
            memory.reset_conversation()
            _clear_widgets()
            st.rerun()
        st.divider()
        st.caption('Chat answers use the financial knowledge library. Analyses include AI explanations grounded in their metrics and retrieved knowledge.')
        st.caption('Identifying report fields are removed before sending analysis context to OpenAI; financial values are retained.')
        with st.expander('Preferences & user memory'):
            st.caption('Currency: USD')
            risk = st.selectbox('Risk preference', ['low', 'moderate', 'high'],
                                index=['low', 'moderate', 'high'].index(memory.profile.risk_preference), key='work_risk')
            if st.button('Save preferences', disabled=memory.profile_key == 'guest'):
                saved = _action(lambda: memory.save_preferences(currency='USD', risk_preference=risk), success='Preferences saved.')
                if saved is not None:
                    st.rerun()
            st.caption('Session memory: this chat, uploads, selected inputs and reports. New conversation clears them.')
            st.caption('User memory: saved preferences, plus holdings and goals only when you click Save. These persist across restarts.')
            if memory.profile.saved_holdings_csv:
                st.write('Saved holdings: ' + (memory.profile.saved_holdings_name or 'Holdings'))
                if st.button('Load saved holdings'):
                    if _action(memory.restore_holdings) is not None:
                        _refresh(memory)
            if memory.profile.saved_plan:
                st.write(f'Saved goals: {len(memory.profile.saved_plan.goals)}')
                if st.button('Load saved goals'):
                    if _action(memory.restore_plan) is not None:
                        _refresh(memory)
            if st.button('Forget saved financial inputs', disabled=memory.profile_key == 'guest'):
                memory.forget_saved_inputs()
                st.session_state['_notice'] = 'Saved holdings and goals removed. The current conversation is unchanged.'
                st.rerun()


def _chat(memory):
    st.subheader('Understand the financial concepts')
    st.caption('Answers use only the financial knowledge library. Use Portfolio, Market, and Goals for analysis of your inputs; their data is not included in this chat.')
    if not memory.messages:
        st.info('Try “What is drawdown?” or “How does diversification reduce concentration risk?”')
    for index, entry in enumerate(memory.messages):
        with st.chat_message(entry['role']):
            if entry['role'] == 'user':
                st.markdown(entry['text'])
            else:
                response = entry['response']
                if response.get('agent'):
                    st.caption({'chat': 'Financial knowledge', 'portfolio': 'Portfolio analysis',
                                'market': 'Market analysis', 'goals': 'Goals planning'}[response['agent']])
                report_response(memory, response, key=f'chat_{index}', show_charts=False)
    text = st.chat_input('Ask Finnie…', max_chars=16000)
    if text:
        with st.spinner('Searching the financial knowledge library…'):
            response = _action(lambda: memory.ask_chat(text))
        if response is not None:
            st.rerun()


def _portfolio(memory):
    st.subheader('Understand what you own')
    st.caption('Upload current holdings. All portfolio values come from your file; this analysis does not reprice them.')
    with st.container(border=True):
        upload = st.file_uploader('Holdings · CSV or Excel', type=['csv', 'xlsx'], key='work_holdings_upload')
        st.caption('Required: ticker, market_value_usd and asset_type or asset_class. USD, long-only holdings; sectors and valuation dates are optional.')
        left, right = st.columns(2)
        left.download_button('Download CSV template', TEMPLATE, file_name='holdings_template.csv', mime='text/csv')
        if right.button('Load example portfolios'):
            path = PROJECT_ROOT / 'src/data/sample_portfolio_holdings.csv'
            if _action(lambda: memory.add_holdings(path.read_bytes(), path.name)) is not None:
                st.session_state['_notice'] = 'Educational sample portfolios loaded. Select one to analyze.'
                _refresh(memory)
        if upload is not None and st.button('Load holdings'):
            if upload.size > MAX_FILE_BYTES:
                st.error('Holdings file must be at most 10 MB.')
            elif _action(lambda: memory.add_holdings(upload.getvalue(), upload.name)) is not None:
                _refresh(memory)
    options = [m for m in memory.service.store.list(memory.thread_id) if m['kind'] == 'portfolio']
    if options:
        ids = [m['id'] for m in options]
        labels = {m['id']: m['name'] for m in options}
        index = ids.index(memory.selected_portfolio) if memory.selected_portfolio in ids else 0
        chosen = st.selectbox('Portfolio', ids, index=index,
                              format_func=labels.get, key='work_portfolio_choice')
        memory.selected_portfolio = chosen
        data = memory.service.store.get(memory.thread_id, chosen, kind='portfolio')
        st.caption(f"{data['name']} · valuation date: {data.get('as_of_date') or 'not supplied'} · {data['input_source']}")
        with st.expander('View supplied holdings'):
            st.dataframe(data['holdings'], hide_index=True, width='stretch')
        left, right = st.columns(2)
        if left.button('Analyze portfolio', type='primary', width='stretch'):
            _run(memory, 'Analyze ' + data['name'], route='portfolio', arguments={'portfolio_id': chosen}, resume=False)
        if right.button('Save holdings to profile', disabled=memory.profile_key == 'guest', width='stretch'):
            if _action(lambda: memory.save_holdings(chosen), success='Selected holdings saved to your local profile.') is not None:
                st.rerun()
    if 'portfolio' in memory.latest:
        st.divider()
        report_response(memory, memory.latest['portfolio'], key='portfolio_result')


def _market(memory):
    st.subheader('Put market moves in context')
    st.caption('U.S.-listed stocks and ETFs in USD, with optional benchmark comparison. No currency conversion, news, or causal explanations are fetched.')
    with st.form('work_market_form', border=True):
        left, right = st.columns(2)
        symbol = left.text_input('Ticker', placeholder='AAPL', max_chars=16, key='work_market_symbol')
        benchmark = right.text_input('Benchmark · optional', placeholder='SPY', max_chars=16, key='work_market_benchmark')
        sessions = st.slider('Trading sessions', min_value=2, max_value=99, value=20, key='work_market_sessions')
        adjusted = st.checkbox('Use adjusted close · requires premium provider access', key='work_market_adjusted')
        st.caption('20 sessions means 20 observed return intervals using 21 closing prices. Prices are end-of-day, not live quotes.')
        submitted = st.form_submit_button('Analyze market', type='primary')
    if submitted:
        _run(memory, 'Analyze ' + (symbol.strip().upper() or 'market'), route='market', resume=False,
             arguments={'symbol': symbol, 'benchmark': benchmark.strip() or None, 'sessions': sessions, 'adjusted': adjusted})
    if 'market' in memory.latest:
        st.divider()
        report_response(memory, memory.latest['market'], key='market_result')


def _goals(memory):
    st.subheader('Plan your goals, one at a time')
    st.caption('Create multiple individual goals, then select one to analyze. Each goal uses its own savings and monthly budget; these analyses do not combine commitments across goals.')
    with st.expander('Example goals'):
        example = st.selectbox('Example goal', ['House deposit', 'Retirement'], key='work_example_goal')
        if st.button('Load example goals'):
            filename = 'sample_retirement_goal.json' if example == 'Retirement' else 'sample_goals_plan.json'
            data = json.loads((PROJECT_ROOT / 'src/data' / filename).read_text())
            if _action(lambda: memory.add_plan(data)) is not None:
                st.session_state['_notice'] = 'Educational example loaded with explicit assumptions.'
                _refresh(memory)
        if example == 'Retirement':
            st.caption('Illustrative retirement accumulation target in today’s dollars. Growth, inflation and fees are assumptions; withdrawals, pensions and retirement-income adequacy are not modeled.')
    all_plans = {m['id']: m for m in memory.service.store.list(memory.thread_id) if m['kind'] == 'goal_plan'}
    plans = [m for m in all_plans.values() if m['parent_id'] is None
             and len(memory.service.store.get(memory.thread_id, m['id'])['goals']) == 1]
    selected = None
    if plans:
        ids = [m['id'] for m in plans]
        root = memory.selected_plan
        while root in all_plans and all_plans[root]['parent_id']:
            root = all_plans[root]['parent_id']
        index = ids.index(root) if root in ids else len(ids) - 1
        labels = {m['id']: f"{n}. {memory.service.store.get(memory.thread_id, m['id'])['goals'][0]['name']}" for n, m in enumerate(plans, 1)}
        selected = st.selectbox('Your goals', ids, index=index,
                               format_func=labels.get, key='work_plan_choice')
        memory.selected_plan = selected
        left, right = st.columns(2)
        if left.button('Analyze selected goal', type='primary'):
            _run(memory, 'Analyze ' + labels[selected], route='goals', arguments={'plan_id': selected}, resume=False)
        if right.button('Save selected goal to profile', disabled=memory.profile_key == 'guest'):
            if _action(lambda: memory.save_plan(selected), success='Selected goal saved to your local profile.') is not None:
                st.rerun()
    with st.expander('Create a goal', expanded=not plans):
        with st.form('work_goal_form'):
            name = st.text_input('Goal name', placeholder='House deposit', max_chars=160, key='work_goal_name')
            left, right = st.columns(2)
            left.caption('All amounts are in USD.')
            as_of = right.date_input('Plan start date', value=date.today(), key='work_goal_asof')
            target = left.number_input('Target amount', min_value=0.01, value=None, step=100.0, key='work_goal_target')
            deadline = right.date_input('Target date', value=None, min_value=date(2000, 1, 1), max_value=date(2100, 12, 31), key='work_goal_deadline')
            basis = st.selectbox('Target expressed in', ['future_dollars', 'today_dollars'],
                                  format_func=lambda v: 'Future money at the deadline' if v == 'future_dollars' else "Today's purchasing power", key='work_goal_basis')
            st.markdown('**Resources for this goal**')
            left, right = st.columns(2)
            savings = left.number_input('Available savings for this goal', min_value=0.0, value=None, step=100.0, key='work_goal_savings')
            budget = right.number_input('Monthly saving budget for this goal', min_value=0.0, value=None, step=50.0, key='work_goal_budget')
            initial = left.number_input('Starting balance assigned to this goal', min_value=0.0, value=None, step=100.0, key='work_goal_initial')
            contribution = right.number_input('Monthly contribution to this goal', min_value=0.0, value=None, step=50.0, key='work_goal_contribution')
            st.markdown('**Your assumptions**')
            left, right = st.columns(2)
            growth = left.number_input('Annual return assumption (%)', min_value=-99.99, max_value=100.0, value=None, step=0.5, key='work_goal_return')
            inflation = right.number_input('Annual inflation assumption (%)', min_value=-99.99, max_value=100.0, value=None, step=0.5, key='work_goal_inflation')
            fee = left.number_input('Annual fee (%)', min_value=0.0, max_value=99.99, value=0.0, step=0.1, key='work_goal_fee')
            risks = ['low', 'moderate', 'high']
            risk = right.selectbox('Risk tolerance for this goal', risks, index=risks.index(memory.profile.risk_preference), key='work_goal_risk')
            flexible_date = st.checkbox('I can move the deadline', key='work_goal_flexible_date')
            flexible_target = st.checkbox('I can reduce the target', key='work_goal_flexible_target')
            st.caption('Enter 0 explicitly for a zero return or inflation assumption. Risk tolerance does not set a return assumption.')
            submitted = st.form_submit_button('Add goal', type='primary')
        if submitted:
            goal = {'id': 'goal', 'name': name, 'target_amount': target, 'target_date': deadline.isoformat() if deadline else None,
                    'target_basis': basis, 'starting_balance': initial, 'monthly_contribution': contribution,
                    'annual_return_pct': growth, 'annual_inflation_pct': inflation, 'annual_fee_pct': fee,
                    'risk_tolerance': risk, 'deadline_flexible': flexible_date, 'target_flexible': flexible_target}
            plan = {'as_of_date': as_of.isoformat(), 'currency': 'USD', 'available_savings': savings,
                    'monthly_budget': budget, 'goals': [goal]}
            if _action(lambda: memory.add_plan(plan), success='Goal added. Select it above to analyze.') is not None:
                _refresh(memory)
    goal_result = getattr(memory, 'goal_results', {}).get(selected)
    if goal_result:
        st.divider()
        report_response(memory, goal_result, key='goals_result')


def main():
    st.set_page_config(page_title='Finnie · Financial workspace', page_icon='🌿', layout='wide')
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    # Reset only incompatible legacy session state when upgrading the UI.
    if '_memory' not in st.session_state or not hasattr(st.session_state['_memory'], 'chat_service'):
        st.session_state['_memory'] = SessionMemory()
    if '_next_profile' in st.session_state:
        st.session_state['profile_choice'] = st.session_state.pop('_next_profile')
    if '_next_inputs' in st.session_state:
        for key in st.session_state.pop('_next_inputs'):
            st.session_state.pop(key, None)
    memory = st.session_state['_memory']
    _sidebar(memory)
    st.title('A clearer view of your money.')
    st.caption('Explore your holdings, understand markets, and make progress toward your goals.')
    if '_notice' in st.session_state:
        st.success(st.session_state.pop('_notice'))
    chat, portfolio, market, goals = st.tabs(['Chat', 'Portfolio', 'Market', 'Goals'], key='work_tabs', on_change='rerun')
    with chat:
        _chat(memory)
    with portfolio:
        _portfolio(memory)
    with market:
        _market(memory)
    with goals:
        _goals(memory)
    st.divider()
    st.caption(
        'Finnie provides general financial information and analysis, not individualized investment, tax, or legal advice. '
        'It does not assess your complete financial circumstances or determine investment suitability. '
        'Portfolio calculations use supplied holdings; market data may be delayed or incomplete. '
        'Goal projections depend on your assumptions and are not forecasts or guarantees. '
        'Investments can lose value. Verify information and consult a qualified professional before making financial decisions.'
    )


if __name__ == '__main__':
    main()
