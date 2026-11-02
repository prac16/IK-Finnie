"""Alpha Vantage daily-history transport, validation, retries, and bounded caching."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from dotenv import load_dotenv
from tools.alpha_vantage import (
    ALPHA_VANTAGE_URL, AlphaVantageError, AlphaVantageTickerTool, _trusted_urlopen,
)


def normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Za-z0-9.-]{1,15}', symbol.strip()):
        raise ValueError('Ticker must contain 1–15 letters, digits, dots or hyphens')
    return symbol.strip().upper()


def normalize_us_symbol(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    if not re.fullmatch(r'[A-Z]{1,6}(?:[.-][AB])?', symbol):
        raise ValueError('Use a supported U.S.-listed stock or ETF ticker in USD; foreign exchange suffixes are not supported')
    return symbol


class MarketDataUnavailable(AlphaVantageError):
    def __init__(self, message: str, *, retryable: bool = False, rate_limited: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.rate_limited = rate_limited


@dataclass(frozen=True)
class DailyBar:
    date: str
    close: float
    volume: int
    adjusted_close: float | None = None
    split_coefficient: float | None = None


@dataclass(frozen=True)
class DailyHistory:
    symbol: str
    bars: tuple[DailyBar, ...]
    price_basis: str
    retrieved_at_utc: str
    source: str = 'alpha_vantage'
    freshness: str = 'fetched'
    cache_age_seconds: float = 0
    attempts: int = 1
    warning: str | None = None
    provider: str = 'alpha_vantage'
    endpoint: str = 'TIME_SERIES_DAILY'
    market_timezone: str | None = None


class AlphaVantageMarketData:
    """No synthetic historical values; unavailable history raises a typed error."""
    def __init__(self, api_key: str | None = None, *, opener: Callable = _trusted_urlopen,
                 clock: Callable = time.monotonic, sleeper: Callable = time.sleep,
                 jitter: Callable = random.uniform, cache_ttl_seconds: float = 300,
                 max_stale_seconds: float = 86400, timeout_seconds: float = 10):
        for name, value in (('cache TTL', cache_ttl_seconds), ('maximum stale age', max_stale_seconds), ('timeout', timeout_seconds)):
            if not math.isfinite(value) or value < 0 or (name == 'timeout' and value == 0):
                raise ValueError(f'Invalid {name}')
        load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
        self.api_key = api_key if api_key is not None else os.getenv('ALPHA_VANTAGE_API_KEY')
        self._opener, self._clock, self._sleep, self._jitter = opener, clock, sleeper, jitter
        self.cache_ttl_seconds, self.max_stale_seconds = cache_ttl_seconds, max_stale_seconds
        self.timeout_seconds = timeout_seconds
        self._cache: OrderedDict[tuple[str, bool], tuple[DailyHistory, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._quotes = AlphaVantageTickerTool(api_key=self.api_key, opener=opener, clock=clock, sleeper=sleeper, jitter=jitter)

    def get_quote(self, symbol: str) -> dict:
        return self._quotes.get_quote(normalize_symbol(symbol))

    def get_history(self, symbol: str, *, adjusted: bool = False) -> DailyHistory:
        symbol = normalize_symbol(symbol)
        key = (symbol, adjusted)
        with self._lock:
            cached = self._cache.get(key)
            if cached:
                self._cache.move_to_end(key)
            cooldown = self._cooldown_until
        now = self._clock()
        age = max(0, now - cached[1]) if cached else 0
        if cached and age < self.cache_ttl_seconds:
            return replace(cached[0], source='cache', freshness='cached', cache_age_seconds=age, attempts=0)
        error = MarketDataUnavailable('ALPHA_VANTAGE_API_KEY is not configured')
        attempts = 0
        if now < cooldown:
            error = MarketDataUnavailable('Alpha Vantage rate-limit cooldown is active', rate_limited=True)
        elif self.api_key:
            for attempt in range(3):
                attempts = attempt + 1
                try:
                    history = replace(self._fetch(symbol, adjusted), attempts=attempts)
                    with self._lock:
                        self._cache[key] = (history, self._clock())
                        self._cache.move_to_end(key)
                        while len(self._cache) > 32:
                            self._cache.popitem(last=False)
                    return history
                except MarketDataUnavailable as exc:
                    error = exc
                    if not exc.retryable or attempt == 2:
                        break
                    self._sleep(self._jitter(0, 2.0 ** attempt))
            if error.rate_limited:
                with self._lock:
                    self._cooldown_until = self._clock() + 60
        age = max(0, self._clock() - cached[1]) if cached else 0
        if cached and age <= self.max_stale_seconds:
            return replace(cached[0], source='cache', freshness='stale', cache_age_seconds=age,
                           attempts=attempts, warning=str(error))
        raise error

    def _fetch(self, symbol: str, adjusted: bool) -> DailyHistory:
        endpoint = 'TIME_SERIES_DAILY_ADJUSTED' if adjusted else 'TIME_SERIES_DAILY'
        request = Request(ALPHA_VANTAGE_URL + '?' + urlencode({
            'function': endpoint, 'symbol': symbol, 'outputsize': 'compact', 'apikey': self.api_key,
        }))
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read(5 * 1024 * 1024 + 1)
            if len(raw) > 5 * 1024 * 1024:
                raise MarketDataUnavailable('Alpha Vantage history response is too large')
            payload = json.loads(raw.decode('utf-8'))
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise MarketDataUnavailable(f'Alpha Vantage history HTTP {code}', retryable=code in {408, 429} or code >= 500,
                                        rate_limited=code == 429) from None
        except (URLError, TimeoutError, OSError):
            raise MarketDataUnavailable('Could not connect to Alpha Vantage history endpoint', retryable=True) from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MarketDataUnavailable('Invalid JSON in Alpha Vantage history response', retryable=True) from None
        if not isinstance(payload, dict):
            raise MarketDataUnavailable('Invalid Alpha Vantage history response')
        message = payload.get('Note') or payload.get('Information') or payload.get('Error Message')
        if message:
            text = str(message).lower()
            limited = any(term in text for term in ('rate limit', 'call frequency', 'requests per', 'calls per'))
            reason = 'Alpha Vantage rate limit reached' if limited else 'Alpha Vantage history unavailable (key, entitlement, or ticker rejected)'
            raise MarketDataUnavailable(reason, retryable=limited, rate_limited=limited)
        meta = payload.get('Meta Data')
        rows = payload.get('Time Series (Daily)')
        if not isinstance(meta, dict) or not isinstance(rows, dict) or not rows:
            raise MarketDataUnavailable('Alpha Vantage returned no daily history')
        if str(meta.get('2. Symbol', '')).upper() != symbol:
            raise MarketDataUnavailable('Alpha Vantage returned history for a different ticker')
        bars = []
        try:
            for day, row in rows.items():
                date.fromisoformat(day)
                close = float(row['4. close'])
                volume_value = float(row['6. volume' if adjusted else '5. volume'])
                adj = float(row['5. adjusted close']) if adjusted else None
                split = float(row['8. split coefficient']) if adjusted else None
                if (not math.isfinite(close) or close <= 0 or not math.isfinite(volume_value)
                    or volume_value < 0 or volume_value != int(volume_value)
                    or (adjusted and (not math.isfinite(adj) or adj <= 0 or not math.isfinite(split) or split <= 0))):
                    raise ValueError('Invalid price or volume')
                bars.append(DailyBar(day, close, int(volume_value), adj, split))
        except (KeyError, ValueError, TypeError, OverflowError):
            raise MarketDataUnavailable('Invalid dates, prices, or volumes in daily history') from None
        bars.sort(key=lambda bar: bar.date)
        return DailyHistory(symbol, tuple(bars), 'adjusted_close' if adjusted else 'raw_close',
                            datetime.now(timezone.utc).isoformat(), endpoint=endpoint,
                            market_timezone=meta.get('5. Time Zone'))
