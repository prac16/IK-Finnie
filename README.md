# Finnie

Finnie connects financial Q&A, portfolio analysis, market analysis, and goals
planning through a stateful LangGraph router. Financial reference PDFs live in
`knowledge_base/documents`, separately from sample inputs and runtime artifacts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Populate the API key placeholders in `.env`, then build the PDF index with
`finnie-index-pdfs`. Real credentials, local profiles, caches, generated reports,
and the generated index are excluded from Git.

Watch the [five-minute prototype demo](reports/finnie_demo_5min.mp4) or read its
[transcript](reports/demo/transcript.md). It covers architecture, the four agents,
financial calculations, failure handling, and saved profile inputs surviving a
server restart.

## Conversational assistant

Launch the Streamlit workspace:

```bash
finnie-ui
# Or: streamlit run streamlit_app.py
```

The UI has Chat, Portfolio, Market, and Goals tabs, CSV/Excel holdings upload,
goal forms, charts, reports, and direct source-PDF hyperlinks. Session memory keeps
the current conversation and inputs across reruns. Local user profiles persist
preferences and explicitly saved holdings/goals across restarts. Unsaved UI
inputs and reports stay in memory. See [UI and memory](docs/streamlit_ui.md).

UI knowledge retrieval and LLM explanations are always enabled. Chat is a
dedicated knowledge-base Q&A conversation with its own state; it cannot access
Portfolio, Market, or Goals inputs/reports or route requests to those agents.
Analyses run from their dedicated tabs. Identifying report metadata is removed
at the [LLM privacy boundary](docs/llm_privacy.md) before external requests.

```bash
finnie-chat
finnie-chat --holdings src/data/sample_portfolio_holdings.csv "Analyze my holdings"
finnie-chat --plan src/data/sample_goals_plan.json "Can I reach my goal?"
finnie-chat "Analyze AAPL for 20 sessions against SPY"
```

The graph routes to one specialist, validates inputs, pauses for missing values,
and remembers selected inputs and reports for follow-ups within the running
process. Clear routes avoid a routing LLM call; ambiguous requests use at most
one structured router call. The specialist generates the explanation; the
orchestrator adds no second narrative call.

Use `--no-llm --no-rag` for local portfolio/goals calculations. `--no-llm` alone
allows retrieved Q&A excerpts without a generated answer. Market analysis still
uses Alpha Vantage. `.env` supplies the existing API keys; no new key is needed.
Memory is process-local and uploads/reports are stored under gitignored
`runtime/artifacts/`. Combined two-agent workflows remain a later increment.
See [the orchestrator walkthrough](docs/orchestrator.md) for clarification,
structured goal edits, the service API, metrics, and a multi-turn demo.

## Build the PDF index

```bash
finnie-index-pdfs
```

The command reads PDFs from `knowledge_base/documents` and persists the `finnie_pdfs` collection
under `knowledge_base/index/`. Chroma's default `all-MiniLM-L6-v2` embedding model is
downloaded on the first run. Re-running the command upserts chunks with stable
IDs, avoiding duplicates for matching pages and chunks.

Options are available for non-default locations or chunking:

```bash
finnie-index-pdfs --help
```

## Ask questions with linked references

The Q&A agent retrieves relevant chunks from Chroma, uses the OpenAI API to
draft an answer from those excerpts, and adds links to the cited PDF pages. Set
your API key in the environment (do not commit it):

```bash
export OPENAI_API_KEY="your-api-key"
```

Ask a single question:

```bash
finnie-ask "What factors affect investment risk?"
```

Run an interactive session by omitting the question. The default model is
`gpt-4o-mini`; set `OPENAI_MODEL` or pass `--model` to choose another OpenAI
model. Customer questions and retrieved excerpts are sent to OpenAI for
generation.

## Portfolio analysis agent

The local portfolio analysis agent reads sample profiles and holdings from
`src/data/15_sample_investment_portfolios.xlsx`. It calculates asset allocation,
position concentration, effective holding count, and illustrative concentration
scenarios, and screens for known index overlap and goal-related considerations.
Run it for all sample portfolios or a single portfolio:

```bash
finnie-analyze-portfolio
finnie-analyze-portfolio "Tech Heavy"
finnie-analyze-portfolio 6
```

Screens and rebalancing scenarios are educational thresholds, not universal
recommendations. Results identify when source data is insufficient: this sample
does not provide fund look-through holdings, account-by-account placement,
investor tax rates, yields, or return history. The agent does not calculate
income adequacy, tax-equivalent yield, correlations, or historical risk.

An Alpha Vantage ticker quote tool is implemented in
`src/tools/alpha_vantage.py` for future market-analysis agent integration. It is
not wired into `finnie-ask`. Configure an Alpha Vantage API key when integrating
the tool:

```bash
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-api-key"
```

Quotes are cached in memory for 60 seconds by ticker. Responses include the
latest trading day, retrieval time, cache age, and a freshness status. If
Alpha Vantage fails after a quote has been cached, the tool can return that
quote as stale and includes the reason; without a cached value, it reports the
dummy defaults instead (see retry and fallback behavior below).

## Retrieval: filter, rank, select, measure

During ingestion, each page is split into overlapping chunks (default: 1,000
characters, 150-character overlap). Chroma embeds each chunk with its default
`all-MiniLM-L6-v2` model and stores its text, source, page, and category.
At query time, the same embedding model embeds the question. Chroma searches
for nearby vectors using cosine distance.

The retrieval pipeline in `src/rag/retrieval.py`:

1. Applies an optional category metadata filter **before** vector search.
2. Requests `candidate_k` nearest chunks (default 20).
3. Removes unusable chunks and sorts candidates by ascending cosine distance.
   Chroma already returns distance order; this makes ranking explicit, without
   adding a second model or changing the relevance signal.
4. Takes the best `top_k` chunks (default 5) for the answer prompt.
5. Returns retrieval metrics alongside the answer, including on empty retrieval.

```bash
finnie-ask "How does tax-loss harvesting work?" --category taxes --candidate-k 20 --top-k 5 --metrics
```

Categories: `investing`, `taxes`, `trading`, `insurance`, `personal_finance`.
These are single document-level labels, inferred from filenames rather than PDF
contents. Investing is the fallback. A mixed-topic PDF receives one label, so a
filter can exclude useful passages. Omit `--category` to search everything.
Override labels with a JSON object mapping exact PDF filenames to categories:

```json
{"Financial planner - Bogleheads.pdf": "personal_finance"}
```

New ingestion writes categories automatically. For an older index, update only
metadata without re-embedding:

```bash
finnie-index-pdfs --categories-only
finnie-index-pdfs --categories-only --category-map categories.json
```

`--category-map` also works during normal ingestion. Metadata-only updates retain
existing valid labels unless explicitly overridden. Unlabeled chunks in an
older index will not match a category filter until updated.

`--metrics` prints JSON with requested/returned candidate counts, invalid
candidate count, selected count, unique source count, retrieval time in
milliseconds, selected distances (best first), and mean selected similarity.
Timing includes query embedding/search and local ranking; it excludes answer
generation. For this cosine index, similarity = 1 - distance. Similarity is not
a confidence probability, and a top-ranked chunk can still be irrelevant.
There is no relevance cutoff in this step. Precision@k, recall@k, and ranking
quality require a labeled set of questions and relevant chunks; these runtime
diagnostics do not measure those quality metrics.


## Quote retries and fallback sources

For a cache miss or expired cache entry, the Alpha Vantage tool makes at most
three attempts: the initial call plus two retries. Between attempts it uses
exponential backoff with full jitter: a random delay from 0–1 seconds, then
0–2 seconds. Each network attempt has a 10-second timeout. Failures reported as
`AlphaVantageError` (including rate limits and malformed responses) are retried.
Invalid ticker arguments are rejected immediately. A missing API key skips
network attempts and uses the fallback path directly.

After retries are exhausted, the tool prefers a cached real quote, marked
`freshness: stale` with a warning. If none exists, it reads `src/tools/dummy_quotes.json`.
The file contains per-ticker defaults plus a generic default for other tickers.
The requested ticker is retained in `symbol`. Defaults are zero placeholders,
not financial observations: `is_dummy` is true, `freshness` is `dummy`, trading
and retrieval timestamps are null, and the warning says `SYNTHETIC DEFAULT`.
Dummy values never enter the real-quote cache. A missing/invalid fallback file
raises an error rather than inventing a response.

Every response includes `source` (`alpha_vantage`, `cache`, or `dummy_file`),
`provider` (`alpha_vantage` for real/cached data, null for dummy data), and
`attempts` (network attempts made for this call). Dummy responses also identify
`source_file`. Consumers must check `is_dummy` before displaying or calculating
with a quote. A newly fetched response's `freshness: fresh` describes the fetch,
not necessarily the age of the provider's trading-day data.

A valid cache hit skips the API. After retries end in a rate-limit error, the
60-second cooldown skips further network requests and uses cache/dummy fallback.
Stale cache age is currently unbounded, and the cache is in memory only.

## Portfolio uploads and local RAG reports

Portfolio metrics use only supplied holdings values, without quote APIs, cached
quotes, or dummy values. All 15 samples have the inputs for value, allocation,
and concentration metrics; missing valuation dates remain explicitly unknown.

Start the local upload page (CSV/XLSX, Markdown/JSON downloads):

```bash
.venv/bin/python -m agents.portfolio.upload
```

Open http://127.0.0.1:8765. Required columns are `ticker`, `market_value_usd`, and
`asset_type` or `asset_class`. Optional fields include `portfolio`, `as_of_date`,
`age`, `risk_tolerance`, and `goal`. Use USD values and paste Excel formulas as
values. The page includes a CSV template. Repeated tickers are combined before
concentration calculations; supplied weights must reconcile to market values.

CLI equivalent:

```bash
.venv/bin/python -m agents.portfolio.analysis --holdings /path/to/holdings.xlsx --rag --output reports/portfolio.md
```

Reports combine deterministic metrics and findings with actual retrieved PDF
excerpts and page references. After report assembly, one OpenAI call generates
the explanation from the report, goal, and selected chunks. See
[the implementation walkthrough](docs/portfolio_agent.md) for the schemas,
formulas, validation, components, and verification details.

Sample holdings now include `Sector` and `Sector Source`; uploads accept the
same columns. Reports show holding-level sector allocations and sector-data
coverage. Broad funds use `Multi-sector`, non-equity instruments use `Not applicable`,
and missing equity labels use `Unknown`. Labels do not substitute for underlying
fund sector percentages. See the walkthrough for classification sources.

After validation and metric calculation, one vector search combines the stated
goal, holdings, and measured findings. It retrieves 20 candidates and selects
the best five usable chunks. No pre-calculation or follow-up retrieval runs.
The completed structured report then feeds a separate local end-user explanation stage. Explanations include goal relevance, portfolio-level
options, metric provenance, and related PDF references. Portfolio names and
sample focus labels do not trigger unrelated topics; missing goals produce
holdings observations without goal-specific options. The explanation uses
`OPENAI_MODEL` and `OPENAI_API_KEY`, loaded from `.env` without overriding existing
environment values. Portfolio metrics stay local; report context, the goal, and
selected PDF excerpts are sent to OpenAI for explanation. Use `--no-llm` in the
CLI or uncheck AI explanation in the upload page for local-only output. Missing
keys, API errors, refusals, and invalid output produce a clearly labeled local
fallback. The LLM call has a 45-second timeout and no automatic retries.


## Project structure

```text
src/
  agents/
    chat/
      agent.py              # Q&A orchestration, prompts, answer references
      __main__.py           # python -m agents.chat
    portfolio/
      analysis.py           # Validation, metrics, rule-based findings, CLI
      holdings.py           # CSV/XLSX import
      sectors.py            # Sector labels
      report.py             # Analysis → retrieval → explanation coordination
      explanation.py        # Context, prompt, output validation, local fallback
      upload.py             # Local upload page
      __main__.py           # python -m agents.portfolio
    market/
      agent.py              # API data → metrics → optional RAG → explanation
      analysis.py           # Returns, volatility, drawdown and benchmark checks
      explanation.py        # Grounded market explanation and fallback
    goals/
      agent.py              # Multi-goal workflow and report exports
      models.py             # Goal inputs, assumptions and shared funding validation
      projection.py         # Monthly cash flows and contribution solver
      planning.py           # Scenarios, priorities and risk guidance
      explanation.py        # Grounded goals explanation and fallback
  tools/
    alpha_vantage.py        # Quote API, retries, cache, dummy fallback
    market_data.py          # Daily-history transport and cache
    openai_market.py        # Structured market explanation client
    openai_goals.py         # Structured goals explanation client
    openai_generation.py    # Chat OpenAI client
    openai_explanation.py   # Structured explanation OpenAI client and schema
  rag/
    ingest.py               # PDF extraction and indexing
    categories.py           # Document categories
    store.py                # Existing Chroma collection connection
    retrieval.py            # Filtering, ranking, top-k and retrieval metrics
    portfolio.py            # Metric-driven query planning and one portfolio search
    market.py               # Market knowledge retrieval
    goals.py                # Goal-related knowledge retrieval
  data/                     # Sample portfolio and goal inputs only
knowledge_base/
  documents/                # Financial reference PDFs
  index/                    # Generated Chroma index (gitignored)
```

Agents own domain behavior and orchestrate tools; external API requests live in
`tools`, and vector collection access/query logic lives in `rag`. Portfolio
retrieval runs once after metrics, followed by one LLM explanation call.

Module commands:

```bash
.venv/bin/python -m agents.chat --help
.venv/bin/python -m agents.portfolio "Tech Heavy" --rag
.venv/bin/python -m agents.portfolio.upload
```

Installed `finnie-ask`, `finnie-analyze-portfolio`, and `finnie-portfolio-upload`
commands retain their names; their entry points use the new module paths.

## Market analysis agent

Analyze a ticker's daily history, with an optional benchmark, local RAG context,
and one LLM explanation. No news search is used; the Tavily placeholder remains
for a future news agent.

```bash
.venv/bin/python -m agents.market AAPL --sessions 20 --benchmark SPY --rag --output reports/market_aapl.md
.venv/bin/python -m agents.market AAPL --sessions 20 --no-llm --json
```

The installed command is `finnie-analyze-market`. Metrics include period and
daily return, annualized historical volatility, maximum closing-price drawdown,
relative volume, and same-date benchmark return difference. The window uses
trading observations, not calendar days. The default daily endpoint returns
unadjusted closes; dividends are excluded and splits can distort the results.
`--adjusted` explicitly requests the premium adjusted endpoint. Synthetic quote
fallbacks are excluded from analysis. Reports identify their data dates, sources,
missing inputs, and explanation method.

Code lives in `agents/market/` (workflow, calculations, explanation),
`tools/market_data.py` and `tools/openai_market.py` (external APIs), and
`rag/market.py` (one optional retrieval after metrics). See
[the market-agent walkthrough](docs/market_agent.md) for formulas and fallback behavior.

## Financial goals planner

The fourth core agent projects single-currency financial goals with explicit
return/inflation assumptions and a shared savings/contribution budget. It
calculates gaps, required contributions, independent alternatives, and risk-aware
educational guidance. It never double-assigns the declared resources or selects
securities to buy.

```bash
.venv/bin/python -m agents.goals src/data/sample_goals_plan.json --rag --output reports/goals_house.md
.venv/bin/python -m agents.goals src/data/sample_multiple_goals.json --no-llm --json
```

Installed command: `finnie-plan-goals`. One optional post-calculation RAG search
and one final LLM call cover the entire plan. `--no-llm` skips OpenAI. Generated
explanations identify their model or local fallback. Projections remain Python
calculations and are conditional on assumptions, not probabilities or guarantees.
See [the goals-planner walkthrough](docs/goals_planner.md) for input fields,
monthly timing, formulas, risk policy, alternatives, and limitations.
