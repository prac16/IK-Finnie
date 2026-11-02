# Market analysis agent

The first version analyzes one ticker's daily history, optionally against a
benchmark. No Tavily, Serp, news, or internet search is called. The existing
TAVILY_API_KEY placeholder is left untouched for a later news agent.

## Run

```bash
.venv/bin/python -m agents.market AAPL --sessions 20 --benchmark SPY --rag --output reports/market_aapl.md
.venv/bin/python -m agents.market AAPL --sessions 20 --no-llm --json --output reports/market_aapl.json
```

The installed entry point is `finnie-analyze-market`. API keys are loaded from
project `.env` without overriding environment variables. ALPHA_VANTAGE_API_KEY
is needed for data; OPENAI_API_KEY and OPENAI_MODEL configure the optional AI
explanation. --no-llm skips OpenAI. --rag enables one local vector search after
calculation, requesting 20 candidates and selecting five usable passages.

## Components and execution order

1. `agents/market/agent.py` validates ticker, benchmark, and requested sessions.
2. `tools/market_data.py` fetches daily prices and volume. An optional benchmark
   adds one historical-data lookup. Independent provider lookups run sequentially
   to avoid unnecessary API bursts.
3. `agents/market/analysis.py` validates data and calculates metrics in Python.
4. `rag/market.py` optionally retrieves concepts in one post-metric search.
5. A `MarketReport` assembles metrics, findings, provenance, and limitations.
6. `agents/market/explanation.py` builds context from the completed report.
   `tools/openai_market.py` makes one strict-schema OpenAI request, with a
   45-second timeout and no automatic retries. The agent validates metric-field
   references, knowledge IDs, and numeric literals. These checks do not establish
   semantic correctness or citation entailment.
7. Markdown or JSON is printed or written to --output.

Metrics and market data stay authoritative. The LLM explains historical
observations; it must not invent news, reasons for price movements, forecasts,
trade recommendations, or suitability conclusions. Failed or invalid generation
returns a visibly labeled local summary while preserving the metrics.

## Metrics and definitions

The requested window is 2–99 observed return intervals, not calendar days.
The default is 20 intervals, requiring 21 closing observations. A shorter
available history is reported explicitly, with its actual dates and count.

- Period return: last analysis price / first analysis price - 1.
- End-date daily return: last analysis price / previous analysis price - 1.
- Volatility: sample standard deviation of observed session returns multiplied
  by sqrt(252), expressed as a percentage. At least two returns are required.
  A short window is a noisy historical estimate; it is not a future risk bound.
- Maximum drawdown: largest decline from a prior running peak inside the window,
  expressed as a positive magnitude. Closing prices exclude intraday drawdowns.
- Relative volume: final session volume / average volume over the preceding
  observations in the selected window. A zero baseline yields unavailable.
  With adjusted data, known splits disable this raw-volume comparison.
- Benchmark difference: stock return minus benchmark return, in percentage
  points, over the same dates and price basis. A missing/incompatible benchmark
  is omitted, preserving the stock's own analysis. A lagging benchmark can move
  the comparison end date backward; this is explicitly reported. Missing dates
  inside the comparison window are never forward-filled or treated as daily
  observations.

## Price basis and endpoint access

The default is TIME_SERIES_DAILY with outputsize=compact (latest 100 observations).
Its closing prices are raw/unadjusted. Returns exclude dividends and can be
distorted by splits or other corporate actions. All reports state this limitation.

Use --adjusted to explicitly request TIME_SERIES_DAILY_ADJUSTED. This endpoint
requires the relevant Alpha Vantage entitlement. Returns use adjusted closes;
raw prices are never silently substituted into an adjusted analysis. Prices are
in the instrument's quote currency; the endpoint does not supply verified
currency metadata. Benchmark comparisons assume comparable currencies and do
not perform conversion. The 252-session convention is an assumption, not an
exchange-specific calendar calculation.

Provider documentation: https://www.alphavantage.co/documentation/

## Errors, provenance, and fallbacks

Transport requests verify TLS with the existing trusted certifi bundle. Transient
network failures, HTTP 408/429/5xx, and rate limits receive up to two retries with
full jitter (0–1 seconds, then 0–2 seconds). Permanent entitlement/schema/ticker
errors are not retried. Exhausted rate limits activate a 60-second cooldown.

History is cached in memory by ticker and price basis: 32 entries maximum,
five-minute normal TTL, and a maximum cache age of 24 hours for failure fallback.
Each dataset includes provider, endpoint, retrieval timestamp, actual latest data
date, source, cache age, attempts, and freshness. Recently fetched data can still
have an old trading date; reports flag histories ending over seven calendar
days ago. This heuristic does not replace exchange-calendar validation.

If real history is unavailable, the existing quote tool is attempted for a dated
quote-only partial report. Historical metrics and LLM generation are skipped in
that mode. Dummy quotes, invalid prices, and excessively old cached quotes never
become market facts or metric inputs. With no usable real data, status is
unavailable and the CLI exits with code 1. Invalid requests exit through argparse.
A historical report can be partial when the requested window or benchmark is
unavailable. Retrieval failure leaves market metrics intact.

## Verification

Tests use hand-checkable returns and drawdowns, aligned/misaligned benchmarks,
raw versus adjusted prices, split-sensitive volume, malformed provider payloads,
retry timing, cache expiration, dummy exclusion, single RAG and LLM calls,
structured output validation, and module boundaries. No automated unit test
uses real API keys. A live AAPL/SPY smoke test is saved in reports/market_aapl.md
and reports/market_aapl.json with actual dates and explanation-source metadata.
