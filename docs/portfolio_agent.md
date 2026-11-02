# Portfolio agent implementation walkthrough

This version analyzes USD, long-only holdings from CSV/XLSX. All portfolio
numbers come from the file. It never imports or calls the Alpha Vantage tool,
its quote cache, or its dummy defaults. “Current metrics” means metrics
recomputed from uploaded values, as of the user's supplied valuation date.

## 1. Input contract

`src/agents/portfolio/holdings.py` accepts uploaded bytes plus the filename. It
never uses the filename to choose an on-disk destination. CSV is UTF-8; Excel
is XLSX. XLS files must first be saved as XLSX. The Excel `Holdings` sheet is
preferred, otherwise the first sheet is used. Existing sample workbooks can
also include their `Portfolios` profile sheet.

Required fields:

| Field | Meaning |
|---|---|
| ticker | Instrument identifier; repeated tickers are combined for analysis |
| market_value_usd | Nonnegative finite market value in USD |
| asset_type or asset_class | Classification for allocation calculations |

Title-style headers such as `Market Value USD` and `Weight %` are also accepted.
Optional fields include `portfolio`, `portfolio_id`, `as_of_date` (YYYY-MM-DD),
`age`, `risk_tolerance`, `goal`, and `analysis_focus`. Repeat profile/date fields
consistently for all holdings in a portfolio. Optional `weight_pct` is in
percentage points (60 means 60%), and must agree with market value within 0.1
percentage points. Optional currency must be USD. No foreign exchange occurs.

For instance:

```csv
ticker,asset_type,market_value_usd,portfolio,as_of_date
AAPL,Stock,6000,My portfolio,2026-10-30
BND,Bond ETF,4000,My portfolio,2026-10-30
```

This is an illustration, not today's market data. Supply the actual date of your
holdings. A missing date is shown as unknown; missing age, risk tolerance, and
goals are not inferred.

Supported asset classes: US equity, International equity, Global equity,
Fixed income, Cash and cash equivalents, Real estate, Commodities, Crypto assets.
The existing instrument-type mapping supports the sample ETFs; `Stock` maps to
US equity, so a non-US stock requires an explicit `asset_class`. Unknown
classifications fail validation rather than being silently guessed.

## 2. Validate and normalize

The importer checks file size, row count, required columns, numeric finiteness,
nonnegative values, positive total value, consistent portfolio metadata, and
dates. It rejects duplicate header names, non-USD currency, and formulas in
Excel holdings. Paste formula results as values before uploading. A 10 MB
compressed file limit and 50 MB expanded XLSX limit apply.

The engine validates the existing sample data too. Repeated tickers are summed
before calculating concentration; conflicting classifications for the same
ticker are rejected. Zero-value positions are excluded from concentration.

## 3. Calculate metrics in Python

`src/agents/portfolio/analysis.py` owns deterministic calculations:

- Total value = sum of supplied holding market values.
- Position weight = ticker market value / portfolio total.
- Allocation = sum of weights within each asset class.
- Largest position and top-three position weights.
- HHI = sum of squared position weights, using fractions such as 0.60.
- Effective holding count = 1 / HHI.
- Known technology-stock and crypto exposure, using the existing explicit rules.

For a $6,000 / $4,000 two-position portfolio, weights are 60% / 40%, HHI is
0.60² + 0.40² = 0.52, and effective holding count is 1 / 0.52 ≈ 1.92.
This measures instrument concentration, not underlying ETF constituent risk.

Existing screening rules identify large positions, known index overlap,
retirement considerations, and other profile-specific observations. These are
educational screens rather than a suitability decision or an instruction to trade.

## 4. Retrieve financial context

`src/agents/portfolio/report.py` coordinates analysis and the local PDF index.
After metrics are calculated, the coordinator combines the stated goal, actual
holdings, and relevant measured findings into one query. There is exactly one
vector search per report: 20 candidates ranked by cosine distance, then the
best five usable chunks. Tax goals use the taxes category; other goals use
investing. Every selected passage carries its PDF name, page, and retrieval
metrics. The upload page can open the original PDF at that page.

The report combines calculated results, rule-based findings, and retrieved
evidence. One LLM call then explains the completed report in the context of the
goal and selected chunks. Python owns every calculation. The application does
not claim that similarity proves relevance or that retrieved tax material is current.

A missing/failing RAG index is explicitly reported; valid calculations remain
available. Uploaded holdings are not embedded. When AI explanation is enabled, holdings,
report metrics/findings, the goal, and selected PDF chunks are sent to OpenAI.
The embedding model must be available locally; Chroma may download it on its
first use if it has not previously been cached.

## 5. Upload and export

Start the local page:

```bash
cd /Users/prachi/ik/Finnie
.venv/bin/python -m agents.portfolio.upload
```

Open http://127.0.0.1:8765, choose CSV/XLSX, and select Analyze holdings. Download
the assembled Markdown report or structured JSON. The server processes uploads
in memory, binds to localhost, and requires its page's upload token on POST.
This is a local development tool, not a deployed multi-user service.

The CLI uses the same loader and report builder:

```bash
.venv/bin/python -m agents.portfolio.analysis --holdings /path/to/holdings.csv --rag --output reports/my_portfolio.md
.venv/bin/python -m agents.portfolio.analysis "Tech Heavy" --rag
```

The `finnie-portfolio-upload` entry point is available after reinstalling the
editable package. No new runtime dependencies were added for this upload page.

## 6. Sample audit and missing inputs

All 15 original sample portfolios already have the inputs needed for the
metrics above. Both sample CSV and XLSX reconcile to the same totals and
allocations; no market values, share prices, or quantities were invented.
Valuation dates were not supplied, so they remain unknown in reports.

Historical performance, volatility, correlations, Sharpe ratio, exact ETF overlap,
portfolio yield, tax-equivalent yield, and retirement-income adequacy are not
calculated. These need additional inputs such as return series, dated fund
constituents, yields, tax rates, spending, and other income. Holdings snapshots
alone do not supply those facts. A future extension should define the input
contract for each new metric before supplementing educational fixtures.

Verification covers all 15 samples through both loaders, independently known
concentration calculations, invalid numbers and weights, duplicate tickers,
Excel formulas, profile gaps, mocked RAG failures, upload endpoint behavior,
and real retrieval from the local index. Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Holding-level sectors

Samples now include `Sector` and `Sector Source` in CSV and Excel. Uploads accept
`sector` and `sector_source`. The 11 named sectors use GICS-style terminology;
these are curated educational labels, not a licensed, dated GICS constituent
feed. Taxonomy reference: https://www.spglobal.com/spdji/en/landing/topic/gics/ .
Issuer sector reference (including COIN as Financials):
https://www.ishares.com/uk/professionals/en/products/328618/ishares-blockch .

NVDA/MSFT/AAPL are Information Technology; GOOGL is Communication Services;
JNJ is Health Care; KO is Consumer Staples; TSLA is Consumer Discretionary;
COIN is Financials. XLP/XLU/VNQ receive their sector mandates: Consumer Staples,
Utilities, and Real Estate. A mandate labels the whole holding for reporting;
it is not an exact measurement of the fund's current underlying sector exposure.
Broad equity funds (including QQQ) are Multi-sector. Bonds, cash equivalents,
gold, and crypto funds are Not applicable to the equity-sector taxonomy.

Missing equity labels become Unknown, without ticker-based inference. Sector
percentages use total portfolio value, and the sector buckets including
Multi-sector, Unknown, and Not applicable sum to 100%. Sector coverage uses a
different denominator: specifically labeled equity/real-estate value divided by
all equity/real-estate value. For Tech Heavy, coverage is 100% of the 90% equity
sleeve. Its portfolio allocation is 75% Information Technology, 15% Communication
Services, 10% Not applicable. Technology-related business exposure and formal
Information Technology sector exposure are different concepts.

The JSON field `known_technology_stock_pct` is retained for compatibility but now
means individual stocks explicitly labeled Information Technology. It no longer
uses a hardcoded ticker set. Fund look-through remains out of scope.

## Goal-scoped retrieval and end-user explanation

The pipeline is sequential:

1. Validate and calculate deterministic holdings metrics.
2. Plan one query from the stated goal, normalized holdings, and measured findings.
3. Retrieve once: 20 candidates, best five usable chunks.
4. Assemble the structured report with metrics, findings, and retrieved evidence.
5. Generate the end-user explanation from that completed report.

`rag.portfolio.plan_retrieval` takes a completed `PortfolioReport`. Measured concentration,
sector exposure, and crypto allocation determine query emphasis. Goal-specific
income, retirement, tax, purchasing-power, and capital-preservation terms are
combined in that same query, never executed as separate searches. There is no
pre-calculation retrieval, parallel retrieval branch, or follow-up retrieval.
The tax category is used for tax goals; otherwise the investing category is used.
Portfolio names and sample focus labels do not trigger unrelated topics.

Only after the structured report is assembled does
`agents.portfolio.explanation.explain_report` generate an `EndUserExplanation`.
Findings carry observations, relevance to the stated goal, portfolio-level
options, metric-field names, and related library reference IDs. These reference
IDs point to the single shared retrieved set. The explanation is included in
Markdown, JSON, the upload page, and every CLI report path. It does not recompute
metrics, perform additional retrieval, change calculated values, or introduce
wider life-planning topics. Missing goals produce descriptive observations and
no goal-specific options.

The explanation now uses a single OpenAI call with a strict JSON schema and
no SDK retries. `.env` supplies OPENAI_API_KEY and OPENAI_MODEL unless already
configured in the environment. Context excludes uploaded filenames and sample
focus labels. Generated findings identify their metric fields and supporting
chunk IDs. Validation checks those references, numeric literals against report
context, and explicit unsupported suitability claims. These are bounded checks;
they do not prove semantic correctness or citation entailment.

Missing keys, API failure, refused/incomplete output, or validation failure
produce a visibly labeled local template fallback. Explanations expose method,
model, and fallback reason in JSON and Markdown. Use --no-llm or disable AI
explanation on the upload page for local-only output. The network timeout is
45 seconds; there is no second retrieval or retry generation call.
