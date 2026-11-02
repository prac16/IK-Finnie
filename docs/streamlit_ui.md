# Streamlit UI and memory

Launch from the project directory after installing the updated package:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
finnie-ui
# Equivalent repository entry point:
streamlit run streamlit_app.py
```

The app serves on http://127.0.0.1:8501 by default. It reads the existing `.env`;
no additional API keys are needed. The installed launcher enforces a localhost
default, a 10 MB upload limit, and disables Streamlit usage telemetry. The repo
configuration supplies the color theme. Tested with Streamlit 1.66.1.

## Pages

| Tab | Workflow |
|---|---|
| Chat | Dedicated knowledge-base Q&A, retrieved excerpts and citations; no access to analytical inputs or reports |
| Portfolio | Validate CSV/XLSX, select a portfolio, inspect holdings, calculate metrics, show allocation/sector charts |
| Market | Ticker, optional benchmark, 2–99 trading sessions, explicit adjusted-price choice; observed price chart and risk metrics |
| Goals | Create independent single goals or load examples; select and analyze each separately, with a projection chart and goal-specific report |

Each UI request goes through `WorkflowService` with an explicit route. Chat
always selects Q&A and has its own service, thread, history and report store.
Portfolio, Market and Goals use a separate analytical service. Their uploaded
inputs, reports, messages and pending requests never enter Chat's context.
Chat sends only its current question and retrieved knowledge to the answer model.
JSON typed into Chat is treated as a question, not as analytical field updates.
A new tab analysis replaces any pending analytical request with its complete
form inputs. Input loading and chart rendering do not make provider/model calls.
Reruns preserve the workflow instance and do not automatically repeat analysis.
The active tab and form drafts also survive normal reruns.

The AI and knowledge toggles have been removed. All UI requests enable RAG and
LLM generation, irrespective of legacy profile flags. Chat requires retrieved
evidence; if the index is unavailable it reports that, and if no evidence is
found it abstains. Analytical agents retrieve after calculation and attempt an
LLM explanation with at most one different-model fallback when the report has usable metrics. Retrieval/model failures
remain explicitly labeled; they never fabricate evidence or results. CLI-only
offline flags remain available for development and tests.

[LLM privacy filtering](llm_privacy.md) removes identifying report metadata at
the provider boundary while retaining the financial values needed for analysis.

Goal examples include a house deposit and a retirement accumulation target.
Select Retirement under Example goal and click Load example goals. Its inputs
are $50,000 initial savings, $1,000 monthly contributions, a $1,500 monthly
budget for this goal, and a $500,000 target in today's purchasing power by 2056-11-01, starting
2026-11-01. Annual assumptions are 5% return, 2.5% inflation and 0.25% fees.
Under those assumptions the future target is $1,048,819.24, projected savings
are $979,120.87, and the shortfall is $69,698.37. The required monthly contribution
is $1,089.52. These are an educational example, not forecasts or advice; pensions,
retirement withdrawals and income adequacy are not modeled.

All profile and goals currency inputs are fixed to USD. Non-USD plans and
holdings are rejected rather than relabeled. Market analysis supports the
restricted U.S.-listed ticker format; no currency conversion is performed.

Example buttons load the existing educational sample inputs. The goal form
leaves target/date, funding, return, and inflation assumptions blank. A visible
zero fee is the domain model's existing default. Risk tolerance is separate
from return assumptions. An incomplete submission pauses for missing values;
it does not guess them. The UI has no goal-plan JSON upload or loaded-plan review section. Each goal is analyzed independently; budgets are not combined.

Reports display the existing explanation and its model/fallback label, actual
data dates, limitations and partial-data status. Responses and their references
appear once inline; the redundant full-report section has been removed.
Citation hyperlinks open the source PDF at the cited
page; the browser PDF viewer provides its own download control. Report, metrics
JSON and separate PDF download buttons have been removed. The CSV upload template
remains available in Portfolio. Market charts use observed
prices from the same history and exact metric window; they never request a
second series or construct synthetic history. Goals charts use calculated
monthly schedules and the target at the deadline.

## Two memory scopes

| | Session memory | User memory |
|---|---|---|
| Identity | Separate random chat and analytical thread IDs in one browser session | Opaque local profile ID, selected explicitly |
| Storage | `st.session_state['_memory']`, containing `SessionMemory` and its own workflow/checkpointer | JSON in gitignored `runtime/users/<profile_id>.json` |
| Contents | Separate chat/analysis messages and reports; analytical pending request, selected inputs, uploads, drafts and API/cache objects | Display name, preferred currency/risk; optionally one saved holdings snapshot and one goals plan |
| Writes | During interaction, in memory only | Explicit Create/Save/Forget actions |
| Lifetime | Survives reruns and tab changes; lost on new conversation, profile switch, browser-session reset/refresh, or server restart | Survives browser and server restarts |
| Reset | New conversation clears session context and drafts | Forget saved financial inputs removes saved holdings/goals and retains preferences |

Each Streamlit session creates separate chat and analytical `WorkflowService`
instances, each with its own `InMemorySaver` and `ArtifactStore(persist=False)`.
No service or financial data uses a global
`st.cache_resource`. The existing CLI keeps its disk artifact behavior; the UI
uses the memory-only option so unsaved financial inputs do not become durable
artifact files. Disconnect/reconnection behavior follows Streamlit session
state; a browser reload can create a fresh session. User-profile selection is
manual after a fresh session, rather than a persistent identifying cookie.

Local profiles are for this single-machine MVP, not authenticated accounts.
All local profiles are selectable on this app. Profile switching clears the
previous conversation and widgets before loading another user's preferences.
It does not load saved financial inputs automatically. This avoids silently
using an old holdings valuation or goals budget in a new analysis.

Preferences initialize visible controls. They do not overwrite an uploaded
plan's currency, risk, dates, or economic assumptions. No balances or return
assumptions are learned from assistant prose. Chat transcripts, reports, market
history, retrieved knowledge, and API credentials never enter user-profile JSON.

## Save and restore

1. Create a local profile in the sidebar; Guest remains session-only.
2. Set the risk preference; currency is fixed to USD. Click Save preferences to persist. Legacy
   AI/RAG profile values are retained for file compatibility but do not control
   UI execution.
3. Load and select a portfolio or goals plan. Click Save holdings/goals to profile
   to persist that selected input. Saving replaces the previous snapshot of that
   type; reports and chat remain session-only.
4. In a later session, select the profile and click Load saved holdings/goals.
   Review the valuation date or plan assumptions before running fresh analysis.

Holdings are saved as normalized CSV and revalidated on restore. Goals are saved
as a validated `GoalPlan`. Profile writes are atomic and owner-only, capped at
12 MB, and profile IDs are validated before filesystem access. Files are local
plaintext; API keys remain in `.env`. The app has no cloud profile store,
cross-device login, or chat-recovery database.

New conversation clears live uploads/reports while keeping the selected local
profile. Forget saved financial inputs clears the saved snapshot/plan but leaves
already loaded session inputs available until the conversation resets.

## Files and verification

- `src/ui/app.py`: forms, tabs, sidebar, user interaction.
- `src/ui/controller.py`: session lifecycle and workflow dispatch.
- `src/ui/memory.py`: validated user profile and atomic local persistence.
- `src/ui/render.py`: charts, explanations, reports and source downloads.
- `src/ui/launcher.py`: `finnie-ui` command.
- `tests/test_ui.py`: memory behavior and Streamlit `AppTest` interactions.

```bash
.venv/bin/python -m unittest discover -s tests -q
```

Tests cover local opt-in persistence, fresh-controller restoration, profile and
thread isolation, normalized holdings roundtrips, duplicate uploads, rerun
idempotence, tab retention, incomplete goal inputs, exact sample projections,
market window selection, chat/analysis context isolation, mandatory RAG/LLM
requests, and new-conversation resets.
Provider calls in UI tests use injected mocks; no API keys or live services are
required for the test suite.

Streamlit references: [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state),
[stateful tabs](https://docs.streamlit.io/develop/api-reference/layout/st.tabs),
[resource cache scope](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).
