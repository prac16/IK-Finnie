# Requirements and milestone review

Reviewed against the supplied [problem statement](https://docs.google.com/document/d/15h5e2CjlRp7PLYU-XjF-P8zjOXH0qP-2/edit)
and [milestones](https://docs.google.com/document/d/1fEFA63T99yFN69IYcezxY2pXlXcaIl5F/edit).

The four prioritized agents, LangGraph routing/state, Streamlit tabs/charts,
CSV/XLSX input, explicit local user profiles, categorized Chroma retrieval,
citations, Alpha Vantage integration, caching/retries/fallbacks, and source/data
dates are implemented. The requested UI now isolates knowledge-only Chat from
the analytical tabs and always enables RAG and LLM explanation attempts.
The current suite passes 207 tests; overall statement coverage measured with
`coverage run --source=src -m unittest discover -s tests -q` is 88%, exceeding
the 80% target. The UI's internal Run details panel has been removed; execution
metrics remain available internally for evaluation and troubleshooting.

## Main remaining items

1. **Analytical agent cooperation:** the orchestrator currently executes one
   specialist per request. Implement and demonstrate the planned bounded
   portfolio-to-goals workflow with explicit funding assignments and shared
   output validation. Keep Chat isolated as requested.
2. **Portfolio expense ratios:** the milestones identify expense ratios among
   essential metrics. Holdings do not yet contain fund expense ratios and the
   agent does not calculate portfolio-weighted expenses/annual fund costs.
   Require supplied or sourced fund fees; do not infer them or treat missing
   fees as zero. Existing HHI/effective-holdings metrics already cover basic
   diversification rather than needing a new arbitrary score.
3. **Knowledge-base curation and glossary:** 82 PDFs are present and indexed,
   but a PDF count does not demonstrate coverage of 50–100 curated articles.
   Add a source/content inventory, coverage and recency audit, explicit category
   review, and the requested glossary. Existing categories are broad; document
   how stock/bond/ETF topics are covered.
4. **Acceptance and performance evidence:** automate representative multi-turn
   acceptance scenarios and produce repeatable latency/provider-failure
   benchmarks. Add a labeled retrieval evaluation set for precision/recall and
   assess answer grounding/citations and routing accuracy. Runtime timings and
   similarity scores are present, but they are not quality benchmarks.
5. **Submission documentation/demo:** consolidate the implemented architecture,
   contracts, setup, privacy, known limitations, API/service reference and
   troubleshooting. Record the requested 5–10 minute demonstration of the four
   agents, multi-turn behavior, failure handling and analytical cooperation.
6. **Operational polish:** add sanitized structured logging of outcomes and
   latency, run beginner/mobile UX acceptance checks, and verify installation
   and launch from a clean environment.

## Document differences to resolve explicitly

- The market UI uses historical end-of-day data, with a dated quote fallback.
  A normal quote lookup/display and an agreed freshness definition would better
  address the problem statement's real-time lookup wording. Do not describe the
  existing history as live/intraday data; genuine intraday access depends on the
  selected provider entitlement.
- The milestones recommend a 30-minute cache TTL; current quote/history TTLs
  are 60/300 seconds. This is a recommendation, not an unimplemented cache.
  Document/configure the selected policy and rate-limit tradeoff.
- Chroma instead of FAISS is explicitly permitted. Shared adapter contracts
  rather than a mandatory common base class, and `pyproject.toml` rather than a
  separate requirements file, are implementation choices to explain rather
  than reasons to rewrite working code.
- The original statement asks for six agents. The milestones prioritize four
  well-built core agents, allow news if time permits, and label tax as stretch.
  The four core agents are present; the two scope statements should be called
  out in the final submission.

## Optional / stretch / future

- News synthesizer: not implemented; Tavily placeholder retained as requested.
- Dedicated tax education agent: not implemented; tax concepts can use Q&A RAG.
- MCP server / Claude Desktop integration: optional bonus, not implemented.
- Docker/cloud artifacts: optional; local launch is sufficient.
- Monte Carlo, brokerage execution, native mobile, distributed deployment and
  advanced ML personalization are future directions, not MVP blockers.
