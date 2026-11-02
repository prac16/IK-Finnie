"""Alpha Vantage global-quote tool with bounded caching and freshness metadata."""

from __future__ import annotations

import json
import os
import re
import random
import ssl
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
from openai.types.chat import ChatCompletionToolParam


def _trusted_urlopen(request: Request, timeout: float) -> object:
    """Verify HTTPS using certifi's CA bundle across Python installations."""
    context = ssl.create_default_context(cafile=certifi.where())
    return urlopen(request, timeout=timeout, context=context)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_CACHE_TTL_SECONDS = 60
RATE_LIMIT_COOLDOWN_SECONDS = 60
MAX_CACHE_ENTRIES = 256
MAX_RETRIES = 2
DEFAULT_DUMMY_QUOTES_PATH = Path(__file__).with_name("dummy_quotes.json")
_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9.-]{1,15}\Z")

ALPHA_VANTAGE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_ticker_quote",
        "description": (
            "Get the latest available daily global quote for a publicly traded "
            "stock ticker, including price, change, trading date, source, and data freshness. Dummy defaults are synthetic and must never be presented as market prices. "
            "Use for questions asking for current or recent ticker values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, for example AAPL or MSFT.",
                }
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
}


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage cannot provide a usable quote."""


class AlphaVantageRateLimitError(AlphaVantageError):
    """Raised when the API key has reached its request limit."""


@dataclass(frozen=True)
class _Quote:
    symbol: str
    price: str
    change: str
    change_percent: str
    latest_trading_day: str
    retrieved_at_utc: str
    fetched_monotonic: float


@dataclass(frozen=True)
class _CacheEntry:
    quote: _Quote


class AlphaVantageTickerTool:
    """Fetch global quotes from Alpha Vantage and cache them in-process."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        timeout_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[..., object] = _trusted_urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        dummy_quotes_path: Path = DEFAULT_DUMMY_QUOTES_PATH,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.api_key = api_key if api_key is not None else os.getenv("ALPHA_VANTAGE_API_KEY")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._opener = opener
        self._sleep = sleeper
        self._jitter = jitter
        self._dummy_quotes_path = dummy_quotes_path
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._rate_limited_until = 0.0

    def get_quote(self, symbol: str) -> dict[str, str | float | bool | None]:
        """Return a ticker quote and explicit cache/data freshness details."""
        normalized_symbol = symbol.strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(normalized_symbol):
            raise ValueError("symbol must be 1-15 letters, digits, dots, or hyphens")
        cached = self._get_cached(normalized_symbol)
        now = self._clock()
        if not self.api_key:
            return self._fallback(normalized_symbol, cached, "ALPHA_VANTAGE_API_KEY is not configured.", 0)
        if cached is not None:
            age = max(0.0, now - cached.quote.fetched_monotonic)
            if age < self.cache_ttl_seconds:
                return self._format_quote(cached.quote, age, "cached", None)
        with self._cache_lock:
            rate_limited_until = self._rate_limited_until
        if now < rate_limited_until:
            error = AlphaVantageRateLimitError(
                "Alpha Vantage rate limit cooldown is active; try again later."
            )
            return self._fallback(normalized_symbol, cached, str(error), 0)

        for attempt in range(MAX_RETRIES + 1):
            try:
                quote = self._fetch_quote(normalized_symbol)
            except AlphaVantageError as exc:
                if attempt < MAX_RETRIES:
                    # Full jitter: uniform delay in [0, 1s], then [0, 2s].
                    self._sleep(self._jitter(0.0, 2.0 ** attempt))
                    continue
                if isinstance(exc, AlphaVantageRateLimitError):
                    with self._cache_lock:
                        self._rate_limited_until = self._clock() + RATE_LIMIT_COOLDOWN_SECONDS
                return self._fallback(normalized_symbol, cached, str(exc), attempt + 1)
            self._put_cached(normalized_symbol, quote)
            result = self._format_quote(quote, 0.0, "fresh", None)
            result["attempts"] = attempt + 1
            return result
        raise AssertionError("Unreachable retry state")

    def _fallback(self, symbol: str, cached: _CacheEntry | None, reason: str, attempts: int) -> dict:
        if cached is not None:
            age = max(0.0, self._clock() - cached.quote.fetched_monotonic)
            result = self._format_quote(cached.quote, age, "stale", reason)
            result["attempts"] = attempts
            return result
        try:
            data = json.loads(self._dummy_quotes_path.read_text())
            values = data["tickers"].get(symbol, data["default"])
            for key in ("price", "change", "change_percent"):
                if not isinstance(values[key], str) or not Decimal(values[key].rstrip("%")).is_finite():
                    raise ValueError("Invalid dummy numeric value")
        except (OSError, ValueError, KeyError, TypeError, AttributeError, InvalidOperation) as exc:
            raise AlphaVantageError("Dummy quote file is missing or invalid; no fallback available.") from exc
        # Defaults never enter the real-quote cache or receive a trading timestamp.
        return {
            "symbol": symbol, "price": values["price"], "change": values["change"],
            "change_percent": values["change_percent"], "latest_trading_day": None,
            "retrieved_at_utc": None, "cache_age_seconds": None,
            "freshness": "dummy", "is_stale": True, "is_dummy": True,
            "source": "dummy_file", "source_file": self._dummy_quotes_path.name,
            "provider": None, "attempts": attempts,
            "warning": f"SYNTHETIC DEFAULT for {symbol}; not market data. {reason}",
        }

    def _get_cached(self, symbol: str) -> _CacheEntry | None:
        with self._cache_lock:
            entry = self._cache.get(symbol)
            if entry is not None:
                self._cache.move_to_end(symbol)
            return entry

    def _put_cached(self, symbol: str, quote: _Quote) -> None:
        with self._cache_lock:
            self._cache[symbol] = _CacheEntry(quote=quote)
            self._cache.move_to_end(symbol)
            while len(self._cache) > MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)

    def _fetch_quote(self, symbol: str) -> _Quote:
        query = urlencode(
            {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key,
            }
        )
        request = Request(f"{ALPHA_VANTAGE_URL}?{query}")
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                raise AlphaVantageRateLimitError(
                    "Alpha Vantage rate limit reached (HTTP 429)."
                ) from exc
            raise AlphaVantageError(
                f"Alpha Vantage request failed with HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise AlphaVantageError(
                f"Could not connect to Alpha Vantage: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise AlphaVantageError(f"Alpha Vantage request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AlphaVantageError("Alpha Vantage returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise AlphaVantageError("Alpha Vantage returned an invalid response.")
        message = payload.get("Note") or payload.get("Information")
        if isinstance(message, str) and message:
            if "rate limit" in message.lower() or "call frequency" in message.lower():
                raise AlphaVantageRateLimitError(
                    "Alpha Vantage rate limit reached; try again later."
                )
            raise AlphaVantageError(f"Alpha Vantage could not provide a quote: {message}")
        api_error = payload.get("Error Message")
        if isinstance(api_error, str) and api_error:
            raise AlphaVantageError(f"Alpha Vantage rejected the request: {api_error}")

        global_quote = payload.get("Global Quote")
        if not isinstance(global_quote, dict) or not global_quote:
            raise AlphaVantageError(
                f"Alpha Vantage returned no quote for ticker {symbol}."
            )
        returned_symbol = global_quote.get("01. symbol")
        price = global_quote.get("05. price")
        change = global_quote.get("09. change")
        change_percent = global_quote.get("10. change percent")
        trading_day = global_quote.get("07. latest trading day")
        if not isinstance(price, str) or not price.strip():
            raise AlphaVantageError("Alpha Vantage returned an incomplete quote.")
        if not isinstance(change, str) or not change.strip():
            raise AlphaVantageError("Alpha Vantage returned an incomplete quote.")
        if not isinstance(change_percent, str) or not change_percent.strip():
            raise AlphaVantageError("Alpha Vantage returned an incomplete quote.")
        if not isinstance(trading_day, str) or not trading_day.strip():
            raise AlphaVantageError("Alpha Vantage returned an incomplete quote.")
        if isinstance(returned_symbol, str) and returned_symbol.upper() != symbol:
            raise AlphaVantageError(
                f"Alpha Vantage returned {returned_symbol} for requested ticker {symbol}."
            )
        try:
            for value in (price, change):
                Decimal(value)
            Decimal(change_percent.rstrip("%"))
        except InvalidOperation as exc:
            raise AlphaVantageError("Alpha Vantage returned invalid quote values.") from exc

        return _Quote(
            symbol=symbol,
            price=price,
            change=change,
            change_percent=change_percent,
            latest_trading_day=trading_day,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            fetched_monotonic=self._clock(),
        )

    @staticmethod
    def _format_quote(
        quote: _Quote, age_seconds: float, freshness: str, warning: str | None
    ) -> dict[str, str | float | bool | None]:
        return {
            "source": "alpha_vantage" if freshness == "fresh" else "cache",
            "provider": "alpha_vantage",
            "is_dummy": False,
            "attempts": 0,
            "symbol": quote.symbol,
            "price": quote.price,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "latest_trading_day": quote.latest_trading_day,
            "retrieved_at_utc": quote.retrieved_at_utc,
            "cache_age_seconds": round(age_seconds, 1),
            "freshness": freshness,
            "is_stale": freshness == "stale",
            "warning": warning,
        }

    def __call__(
        self, arguments: dict[str, object]
    ) -> dict[str, str | float | bool | None]:
        symbol = arguments.get("symbol")
        if not isinstance(symbol, str):
            raise ValueError("symbol must be a string")
        if set(arguments) != {"symbol"}:
            raise ValueError("get_ticker_quote accepts only the symbol argument")
        return self.get_quote(symbol)
