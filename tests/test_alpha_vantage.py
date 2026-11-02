import json
import unittest
import ssl
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from unittest.mock import patch, Mock
from urllib.parse import parse_qs, urlparse

from tools.alpha_vantage import (
    AlphaVantageError,
    AlphaVantageRateLimitError,
    AlphaVantageTickerTool,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def quote_payload(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "Global Quote": {
            "01. symbol": symbol,
            "05. price": "201.2500",
            "07. latest trading day": "2026-10-30",
            "09. change": "1.2500",
            "10. change percent": "0.6250%",
        }
    }


class AlphaVantageTickerToolTests(unittest.TestCase):
    @patch("tools.alpha_vantage.urlopen")
    def test_default_opener_verifies_https_with_trusted_certificates(self, opener):
        opener.return_value = FakeResponse(quote_payload())
        quote = AlphaVantageTickerTool(api_key="test-key").get_quote("AAPL")
        context = opener.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)
        self.assertEqual(quote["symbol"], "AAPL")

    def test_fetches_quote_and_uses_cache_until_ttl_expires(self) -> None:
        now = [100.0]
        requests: list[tuple[str, float]] = []

        def opener(request: object, timeout: float) -> FakeResponse:
            requests.append((request.full_url, timeout))
            return FakeResponse(quote_payload())

        tool = AlphaVantageTickerTool(
            api_key="test-key",
            cache_ttl_seconds=60,
            clock=lambda: now[0],
            sleeper=lambda delay: None,
            opener=opener,
        )

        first = tool.get_quote("aapl")
        now[0] += 15
        second = tool.get_quote("AAPL")

        self.assertEqual(len(requests), 1)
        params = parse_qs(urlparse(requests[0][0]).query)
        self.assertEqual(params["function"], ["GLOBAL_QUOTE"])
        self.assertEqual(params["symbol"], ["AAPL"])
        self.assertEqual(params["apikey"], ["test-key"])
        self.assertEqual(first["freshness"], "fresh")
        self.assertEqual(second["freshness"], "cached")
        self.assertEqual(second["cache_age_seconds"], 15.0)
        self.assertEqual(second["latest_trading_day"], "2026-10-30")
        self.assertEqual(second["price"], "201.2500")

    def test_returns_stale_cached_quote_after_api_rate_limit(self) -> None:
        now = [0.0]
        calls = 0

        def opener(request: object, timeout: float) -> FakeResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return FakeResponse(quote_payload())
            return FakeResponse({"Note": "API call frequency rate limit exceeded"})

        tool = AlphaVantageTickerTool(
            api_key="test-key",
            cache_ttl_seconds=1,
            clock=lambda: now[0],
            sleeper=lambda delay: None,
            opener=opener,
        )
        tool.get_quote("AAPL")
        now[0] = 5.0

        stale = tool.get_quote("AAPL")
        stale_during_cooldown = tool.get_quote("AAPL")

        self.assertEqual(stale["freshness"], "stale")
        self.assertTrue(stale["is_stale"])
        self.assertEqual(stale["cache_age_seconds"], 5.0)
        self.assertIn("rate limit", stale["warning"])
        self.assertIn("cooldown", stale_during_cooldown["warning"])
        self.assertEqual(calls, 4)

    def test_rate_limit_without_cache_uses_dummy_then_cooldown(self) -> None:
        def opener(request: object, timeout: float) -> FakeResponse:
            return FakeResponse({"Note": "API call frequency rate limit exceeded"})

        calls = 0

        def count_calls(request: object, timeout: float) -> FakeResponse:
            nonlocal calls
            calls += 1
            return opener(request, timeout)

        tool = AlphaVantageTickerTool(api_key="test-key", opener=count_calls, sleeper=lambda delay: None)

        first = tool.get_quote("AAPL")
        second = tool.get_quote("MSFT")
        self.assertEqual(first["source"], "dummy_file")
        self.assertEqual(first["attempts"], 3)
        self.assertEqual(second["attempts"], 0)
        self.assertIn("cooldown", second["warning"])
        self.assertEqual(calls, 3)

    def test_reports_missing_key_and_rejects_invalid_symbol(self) -> None:
        tool = AlphaVantageTickerTool(api_key="")

        result = tool.get_quote("AAPL")
        self.assertTrue(result["is_dummy"])
        self.assertIn("ALPHA_VANTAGE_API_KEY", result["warning"])
        with self.assertRaises(ValueError):
            AlphaVantageTickerTool(api_key="test-key").get_quote("AAPL&apikey=x")

    def test_retry_recovers_with_exponential_full_jitter(self):
        opener = Mock(side_effect=[TimeoutError(), TimeoutError(), FakeResponse(quote_payload())])
        sleeper = Mock()
        jitter = Mock(side_effect=[0.5, 1.5])
        tool = AlphaVantageTickerTool(api_key="test-key", opener=opener, sleeper=sleeper, jitter=jitter)
        result = tool.get_quote("aapl")
        self.assertEqual(opener.call_count, 3)
        self.assertEqual([call.args for call in jitter.call_args_list], [(0.0, 1.0), (0.0, 2.0)])
        self.assertEqual([call.args[0] for call in sleeper.call_args_list], [0.5, 1.5])
        self.assertEqual(result["source"], "alpha_vantage")
        self.assertEqual(result["attempts"], 3)
        self.assertFalse(result["is_dummy"])
        self.assertEqual(tool.get_quote("AAPL")["source"], "cache")
        self.assertEqual(opener.call_count, 3)

    def test_exhausted_retries_prefer_stale_cache(self):
        now = [0.0]
        opener = Mock(side_effect=[FakeResponse(quote_payload()), TimeoutError(), TimeoutError(), TimeoutError()])
        tool = AlphaVantageTickerTool(api_key="test-key", opener=opener, cache_ttl_seconds=1,
                                     clock=lambda: now[0], sleeper=lambda _: None)
        tool.get_quote("AAPL")
        now[0] = 10
        result = tool.get_quote("AAPL")
        self.assertEqual(result["source"], "cache")
        self.assertEqual(result["freshness"], "stale")
        self.assertFalse(result["is_dummy"])
        self.assertEqual(result["price"], "201.2500")
        self.assertEqual(result["attempts"], 3)

    def test_failures_use_dummy_without_caching_it(self):
        for error in (TimeoutError(), HTTPError("https://example.test", 503, "unavailable", {}, None)):
            with self.subTest(error=type(error).__name__):
                opener = Mock(side_effect=error)
                tool = AlphaVantageTickerTool(api_key="test-key", opener=opener, sleeper=lambda _: None)
                result = tool.get_quote("ZZZZ")
                self.assertEqual(opener.call_count, 3)
                self.assertEqual(result["symbol"], "ZZZZ")
                self.assertEqual(result["source"], "dummy_file")
                self.assertTrue(result["is_dummy"])
                self.assertIsNone(result["latest_trading_day"])
                self.assertIsNone(tool._get_cached("ZZZZ"))
                self.assertIn("SYNTHETIC", result["warning"])

    def test_broken_dummy_file_reports_failure(self):
        with TemporaryDirectory() as directory:
            tool = AlphaVantageTickerTool(api_key="", dummy_quotes_path=Path(directory) / "missing.json")
            with self.assertRaisesRegex(AlphaVantageError, "Dummy quote file"):
                tool.get_quote("AAPL")


if __name__ == "__main__":
    unittest.main()
