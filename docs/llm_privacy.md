# LLM request privacy

All five OpenAI transports apply local filtering immediately before constructing
API request messages. No external model or redaction service is used for this
step, and no extra generation call is added.

- Portfolio explanations allow measured fields, holdings, valuation date, and a
  coarse financial-goal category. Profile/portfolio names and IDs, ages, source
  filenames, and user-supplied sector-source text are excluded.
- Goals explanations replace user-written names and IDs with request-local
  aliases (`Goal A`, `goal_a`). Metric paths and cross-goal references use these
  aliases. Responses are mapped back locally before the existing numerical,
  budget and citation validators run. The alias map is never sent externally.
- Market explanations retain market metrics and provenance; identifying fields
  are excluded. Saved-report follow-up text is reduced to financial topics such
  as drawdown, shortfall, or concentration rather than forwarding raw prose.
- Router requests exclude conversation/report history, artifact names and file
  IDs. They include the current filtered message and minimal selected context;
  goal IDs are aliased and restored locally.
- Retrieved source filenames become neutral reference labels. Citation numbers,
  pages, excerpts and local source links remain usable in reports.
- Chat and other free text receive best-effort local pattern redaction for
  email addresses, common phone/government/account identifiers, labeled birth
  dates, street addresses, paths/URLs and probable full names. The selected UI
  profile name is supplied as a call-scoped private identifier, never as model
  context. These patterns are not a comprehensive named-entity detector.

Every OpenAI completion explicitly sets `store=False`. This setting does not
mean a provider has no operational retention; it is not a substitute for payload
filtering. Credentials remain in client configuration and are not report data.

Financial values, positions, goal amounts, dates, risk inputs, currency, and
projection assumptions remain necessary for explanation and are sent after
filtering. This is data minimization and pseudonymization, not a claim that the
remaining financial data is anonymous or non-sensitive. Original local input
files, saved profiles and downloadable reports retain their original labels.

`tests/test_llm_privacy.py` inspects the actual mocked SDK messages with synthetic
PII, verifies alias restoration and unchanged numbers, and checks that the UI
profile-name redaction scope survives the actual LangGraph chat execution.
