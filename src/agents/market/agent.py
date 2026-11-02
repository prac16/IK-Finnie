"""Single-ticker market workflow: API → metrics → optional RAG → report → LLM."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol
from urllib.parse import quote as urlquote

from agents.market.analysis import MarketMetrics, calculate_metrics, validate_history
from agents.market.explanation import MarketExplanation, explain_market_report
from rag.ingest import DEFAULT_PDF_DIR
from rag.market import retrieve_market_knowledge
from rag.retrieval import QueryCollection, RetrievalResult
from rag.store import local_collection
from tools.alpha_vantage import AlphaVantageError
from tools.market_data import AlphaVantageMarketData, DailyHistory, normalize_us_symbol
from tools.openai_market import MarketExplanationGenerator


class MarketDataProvider(Protocol):
    def get_history(self, symbol: str, *, adjusted: bool = False) -> DailyHistory: ...
    def get_quote(self, symbol: str) -> dict: ...


@dataclass(frozen=True)
class MarketReport:
    symbol: str
    benchmark_symbol: str | None
    requested_sessions: int
    status: str
    price_basis: str
    metrics: MarketMetrics | None
    data_sources: tuple[dict, ...]
    findings: tuple[str, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]
    quote: dict | None = None
    knowledge: RetrievalResult | None = None
    explanation: MarketExplanation | None = None
    observed_prices: tuple[dict, ...] = ()
    currency: str = 'USD'

    def to_dict(self):
        return asdict(self)

    def to_markdown(self):
        lines = [f'# Market analysis: {self.symbol}', f'Status: {self.status}',
                 f'Return basis: {self.price_basis}. Supported U.S.-listed instruments; prices in USD.']
        if self.explanation:
            lines.append(self.explanation.to_markdown())
        if self.metrics:
            m = self.metrics
            lines.extend(['## Calculated metrics', f'Actual window: {m.start_date} to {m.end_date}; {m.return_intervals} return intervals.',
                          '| Metric | Value |', '|---|---:|',
                          f'| Starting price ({self.price_basis}) | {m.starting_price:.4f} |',
                          f'| End-date close ({self.price_basis}) | {m.latest_close:.4f} |',
                          f'| End-date daily return | {m.daily_change_pct:.2f}% |',
                          f'| Period return | {m.period_return_pct:.2f}% |',
                          f'| Annualized volatility (252 sessions) | {m.annualized_volatility_pct:.2f}% |' if m.annualized_volatility_pct is not None else '| Annualized volatility | Unavailable |',
                          f'| Maximum closing-price drawdown | {m.max_drawdown_pct:.2f}% |',
                          f'| End-date volume | {m.latest_volume:,} |',
                          f'| Volume / prior-window average | {m.volume_vs_prior_average:.2f}× |' if m.volume_vs_prior_average is not None else '| Relative volume | Unavailable |'])
            if m.benchmark_return_pct is not None:
                lines.extend([f'| {m.benchmark_symbol} period return | {m.benchmark_return_pct:.2f}% |',
                              f'| Stock minus benchmark | {m.benchmark_difference_pp:.2f} percentage points |'])
        elif self.quote:
            lines.extend(['## Dated quote only', f'Price: {self.quote["price"]}; trading date: {self.quote["latest_trading_day"]}. Historical metrics unavailable.'])
        lines.append('## Data sources')
        for source in self.data_sources:
            lines.append(f'- {source["symbol"]}: {source["provider"]}, {source["endpoint"]}; data through {source["latest_date"]}; retrieved {source["retrieved_at_utc"]}; {source["freshness"]}, source={source["source"]}, cache age={source["cache_age_seconds"]:.1f}s.')
        lines.append('## Findings')
        lines.extend('- '+finding for finding in self.findings)
        if self.knowledge:
            lines.append('## Financial knowledge references')
            for index, chunk in enumerate(self.knowledge.chunks, 1):
                link = urlquote(str(DEFAULT_PDF_DIR / chunk.source), safe='/')
                lines.extend([f'[{index}] [{chunk.source}, p. {chunk.page}]({link}#page={chunk.page})', '> '+chunk.text])
            lines.append(f'Retrieval: {self.knowledge.metrics.retrieval_ms:.1f} ms; {len(self.knowledge.chunks)} selected chunks.')
        lines.append('## Limitations')
        lines.extend('- '+item for item in self.limitations)
        if self.warnings:
            lines.append('## Data availability warnings')
            lines.extend('- '+item for item in self.warnings)
        rendered = []
        for line in lines:
            if rendered and line.startswith('|') and rendered[-1].startswith('|'):
                rendered[-1] += '\n' + line
            else:
                rendered.append(line)
        return '\n\n'.join(rendered)


def _source(history):
    return {k: v for k, v in asdict(history).items() if k != 'bars'} | {'latest_date': max(b.date for b in history.bars)}


class MarketAnalysisAgent:
    def __init__(self, provider: MarketDataProvider | None = None, *, collection: QueryCollection | None = None,
                 use_llm: bool = True, generator: MarketExplanationGenerator | None = None):
        self.provider = provider or AlphaVantageMarketData()
        self.collection, self.use_llm, self.generator = collection, use_llm, generator

    def analyze(self, symbol: str, *, sessions: int = 20, benchmark: str | None = None, adjusted: bool = False) -> MarketReport:
        symbol = normalize_us_symbol(symbol)
        benchmark = normalize_us_symbol(benchmark) if benchmark else None
        if type(sessions) is not int or not 2 <= sessions <= 99:
            raise ValueError('sessions must be between 2 and 99')
        if benchmark == symbol:
            raise ValueError('Choose a benchmark different from the analyzed ticker')
        warnings, sources = [], []
        history = bench_history = metrics = knowledge = quote = None
        try:
            history = self.provider.get_history(symbol, adjusted=adjusted)
            validate_history(history)
            if history.price_basis != ('adjusted_close' if adjusted else 'raw_close'):
                raise ValueError('History price basis does not match request')
            if history.symbol != symbol:
                raise ValueError('History ticker does not match request')
            if history.market_timezone and history.market_timezone not in {'US/Eastern', 'America/New_York'}:
                raise ValueError('Only U.S.-market history is supported in the USD workspace')
            sources.append(_source(history))
        except (AlphaVantageError, ValueError):
            warnings.append('Daily history unavailable or invalid; historical metrics were not calculated.')
            history = None
        if history and benchmark:
            try:
                bench_history = self.provider.get_history(benchmark, adjusted=adjusted)
                validate_history(bench_history)
                if bench_history.symbol != benchmark:
                    raise ValueError('Benchmark ticker does not match request')
                if bench_history.market_timezone and bench_history.market_timezone not in {'US/Eastern', 'America/New_York'}:
                    raise ValueError('Benchmark must use the supported U.S. market')
                sources.append(_source(bench_history))
            except (AlphaVantageError, ValueError):
                bench_history = None
                warnings.append('Benchmark history unavailable or invalid; comparison omitted.')
        if history:
            try:
                metrics, notes = calculate_metrics(history, sessions, bench_history)
            except ValueError:
                if bench_history is None:
                    raise
                metrics, notes = calculate_metrics(history, sessions)
                warnings.append('Benchmark dates or price basis are incompatible; comparison omitted.')
            warnings.extend(notes)
            for series in (history, bench_history):
                if series and series.warning:
                    warnings.append(series.symbol+': '+series.warning)
                if series and (datetime.now(timezone.utc).date() - date.fromisoformat(max(b.date for b in series.bars))).days > 7:
                    warnings.append(series.symbol+': provider history ends more than seven calendar days ago; inspect data date.')
        else:
            try:
                candidate = self.provider.get_quote(symbol)
                price = float(candidate.get('price', 'nan'))
                trading_date = date.fromisoformat(candidate.get('latest_trading_day', ''))
                if (candidate.get('is_dummy') is not False or candidate.get('source') not in {'alpha_vantage', 'cache'}
                    or candidate.get('symbol') != symbol or not math.isfinite(price) or price <= 0
                    or (candidate.get('cache_age_seconds') or 0) > 86400):
                    raise ValueError('Unusable quote')
                quote = candidate
                sources.append({'symbol': symbol, 'provider': 'alpha_vantage', 'endpoint': 'GLOBAL_QUOTE',
                                'latest_date': trading_date.isoformat(), 'retrieved_at_utc': candidate.get('retrieved_at_utc'),
                                'freshness': candidate.get('freshness'), 'source': candidate.get('source'),
                                'cache_age_seconds': candidate.get('cache_age_seconds') or 0})
            except (AlphaVantageError, ValueError, TypeError):
                warnings.append('No usable real quote is available. Synthetic defaults were excluded.')
        basis = history.price_basis if history else ('adjusted_close' if adjusted else 'raw_close')
        limitations = [
            'End-of-day observations, not real-time prices; actual data dates and retrieval times are separate.',
            'Historical volatility and drawdown are window-specific, not forecasts or maximum possible future losses.',
            'Volatility uses sample standard deviation of observed session returns and a 252-session annualization assumption.',
            'Drawdown uses closing prices only; intraday declines can be larger.',
            'No news was fetched; this report cannot explain the causes of price movements.',
        ]
        if basis == 'raw_close':
            limitations.append('Unadjusted price returns exclude dividends; splits and other corporate actions can distort all return-based metrics.')
        else:
            limitations.append('Adjusted-close returns reflect the provider’s corporate-action adjustments; they are not raw market quotes.')
        if benchmark:
            limitations.append('Benchmark comparison uses identical observed dates for supported U.S.-listed USD instruments; currency conversion is not performed.')
        findings = []
        if metrics:
            findings.append(f'{symbol} returned {metrics.period_return_pct:.2f}% over {metrics.return_intervals} observed return intervals.')
            findings.append(f'Maximum closing-price drawdown within that window was {metrics.max_drawdown_pct:.2f}%.')
            if metrics.benchmark_difference_pp is not None:
                findings.append(f'Return minus {benchmark}: {metrics.benchmark_difference_pp:.2f} percentage points.')
            if self.collection is not None:
                try:
                    knowledge = retrieve_market_knowledge(self.collection, symbol, asdict(metrics))
                    if not knowledge.chunks:
                        warnings.append('No usable financial knowledge excerpts were retrieved.')
                except Exception:
                    warnings.append('Local financial knowledge retrieval unavailable; market metrics remain available.')
        status = 'unavailable' if metrics is None and quote is None else ('partial' if metrics is None or any('omitted' in w or 'only ' in w for w in warnings) else 'complete')
        report = MarketReport(symbol, benchmark, sessions, status, basis, metrics, tuple(sources), tuple(findings),
                              tuple(limitations), tuple(warnings), quote, knowledge,
                              observed_prices=tuple({'date': bar.date, 'close': bar.close} for bar in history.bars
                                  if metrics.start_date <= bar.date <= metrics.end_date) if metrics and history else ())
        return replace(report, explanation=explain_market_report(report, use_llm=self.use_llm, generator=self.generator))


def main():
    parser = argparse.ArgumentParser(description='Analyze one ticker using Alpha Vantage daily prices; no news search.')
    parser.add_argument('ticker')
    parser.add_argument('--sessions', type=int, default=20, help='Return intervals, 2–99; defaults to 20 trading sessions')
    parser.add_argument('--benchmark', help='Optional comparison ticker, e.g. SPY')
    parser.add_argument('--adjusted', action='store_true', help='Use premium daily adjusted endpoint; never silently switch return basis')
    parser.add_argument('--rag', action='store_true', help='Retrieve financial concepts once, after metrics')
    parser.add_argument('--no-llm', action='store_true', help='Keep output local; skip OpenAI explanation')
    parser.add_argument('--json', action='store_true', help='Output structured JSON')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    collection, rag_warning = None, None
    if args.rag:
        try:
            collection = local_collection()
        except Exception:
            rag_warning = 'Local RAG index unavailable; no knowledge excerpts attached.'
    agent = MarketAnalysisAgent(collection=collection, use_llm=not args.no_llm)
    try:
        report = agent.analyze(args.ticker, sessions=args.sessions, benchmark=args.benchmark, adjusted=args.adjusted)
    except ValueError as exc:
        parser.error(str(exc))
    if rag_warning:
        report = replace(report, warnings=(*report.warnings, rag_warning))
    text = json.dumps(report.to_dict(), indent=2, allow_nan=False) if args.json else report.to_markdown()
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
    if report.status == 'unavailable':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
