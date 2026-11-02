"""Deterministic close-to-close price analysis. No provider or model calls."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from statistics import mean, stdev

from tools.market_data import DailyBar, DailyHistory


@dataclass(frozen=True)
class MarketMetrics:
    requested_sessions: int
    return_intervals: int
    start_date: str
    end_date: str
    starting_price: float
    latest_close: float
    daily_change_pct: float | None
    period_return_pct: float
    annualized_volatility_pct: float | None
    max_drawdown_pct: float
    latest_volume: int
    volume_vs_prior_average: float | None
    benchmark_symbol: str | None = None
    benchmark_return_pct: float | None = None
    benchmark_difference_pp: float | None = None


def validate_history(history: DailyHistory) -> None:
    if history.source not in {'alpha_vantage', 'cache'} or history.provider != 'alpha_vantage':
        raise ValueError('Historical metrics require real provider data, not synthetic values')
    if history.price_basis not in {'raw_close', 'adjusted_close'} or history.freshness not in {'fetched', 'cached', 'stale'}:
        raise ValueError('Unsupported price basis or freshness status')
    seen = set()
    for bar in history.bars:
        date.fromisoformat(bar.date)
        if bar.date in seen:
            raise ValueError('Duplicate dates in historical data')
        seen.add(bar.date)
        value = bar.adjusted_close if history.price_basis == 'adjusted_close' else bar.close
        if value is None or not math.isfinite(value) or value <= 0 or not math.isfinite(bar.close) or bar.close <= 0:
            raise ValueError('Historical prices must be finite and positive')
        if not isinstance(bar.volume, int) or bar.volume < 0:
            raise ValueError('Historical volumes must be nonnegative integers')
    if len(seen) < 2:
        raise ValueError('At least two closing prices are required')


def calculate_metrics(history: DailyHistory, sessions: int = 20,
                      benchmark: DailyHistory | None = None) -> tuple[MarketMetrics, tuple[str, ...]]:
    if type(sessions) is not int or not 2 <= sessions <= 99:
        raise ValueError('sessions must be between 2 and 99 with compact daily history')
    validate_history(history)
    bars = sorted(history.bars, key=lambda bar: bar.date)
    warnings = []
    if benchmark:
        validate_history(benchmark)
        if benchmark.price_basis != history.price_basis:
            raise ValueError('Stock and benchmark must use the same price basis')
        benchmark_by_date = {bar.date: bar for bar in benchmark.bars}
        common = [bar for bar in bars if bar.date in benchmark_by_date]
        # Never call a multi-session jump a daily return or silently fill a gap.
        if len(common) < 2:
            raise ValueError('Stock and benchmark have fewer than two common trading dates')
        if common[-1].date != bars[-1].date:
            warnings.append('Benchmark lacks the latest stock date; comparison ends at the latest common date.')
        end = common[-1].date
        stock_to_end = [bar for bar in bars if bar.date <= end]
        selected = stock_to_end[-(sessions + 1):]
        if any(bar.date not in benchmark_by_date for bar in selected):
            raise ValueError('Benchmark is missing dates inside the selected stock window')
        benchmark_window = [benchmark_by_date[bar.date] for bar in selected]
        benchmark_selected_dates = [bar.date for bar in benchmark.bars if selected[0].date <= bar.date <= end]
        if set(benchmark_selected_dates) != {bar.date for bar in selected}:
            raise ValueError('Stock and benchmark trading calendars differ inside the window')
    else:
        selected = bars[-(sessions + 1):]
        benchmark_window = []
    if len(selected) < sessions + 1:
        warnings.append(f'Requested {sessions} return intervals; only {len(selected)-1} are available. Actual dates are shown.')

    def prices(items):
        return [bar.adjusted_close if history.price_basis == 'adjusted_close' else bar.close for bar in items]

    values = prices(selected)
    returns = [right / left - 1 for left, right in zip(values, values[1:])]
    if not all(math.isfinite(r) for r in returns):
        raise ValueError("Historical prices produce nonfinite returns")
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, 1 - value / peak)
    prior_volumes = [bar.volume for bar in selected[:-1]]
    avg_volume = mean(prior_volumes)
    benchmark_return = (prices(benchmark_window)[-1] / prices(benchmark_window)[0] - 1) * 100 if benchmark_window else None
    period_return = (values[-1] / values[0] - 1) * 100
    if not math.isfinite(period_return):
        raise ValueError('Historical prices produce a nonfinite period return')
    split_in_window = history.price_basis == 'adjusted_close' and any(
        bar.split_coefficient is not None and bar.split_coefficient != 1 for bar in selected
    )
    if split_in_window:
        warnings.append('A split occurs in the window; raw-volume ratio is omitted because volume is not split-adjusted.')
    # Daily move is based on the source's latest adjacent bars, not a benchmark-induced gap.
    volatility = stdev(returns) * math.sqrt(252) * 100 if len(returns) >= 2 else None
    return MarketMetrics(
        sessions, len(returns), selected[0].date, selected[-1].date, values[0], values[-1],
        returns[-1] * 100, period_return, volatility, worst * 100, selected[-1].volume,
        selected[-1].volume / avg_volume if avg_volume > 0 and not split_in_window else None,
        benchmark.symbol if benchmark else None, benchmark_return,
        period_return - benchmark_return if benchmark_return is not None else None,
    ), tuple(warnings)
