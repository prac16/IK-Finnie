# Financial goals planner

This is an educational accumulation planner: it calculates whether assigned
savings and contributions reach stated targets under explicit assumptions. It
compares independent alternatives and explains the results. It does not select
securities, prescribe asset weights, trade, fetch market/news data, or run Monte
Carlo. The Tavily placeholder and existing agents are unchanged.

## Run

```bash
.venv/bin/finnie-plan-goals src/data/sample_goals_plan.json --rag --output reports/goals_house.md
.venv/bin/python -m agents.goals src/data/sample_multiple_goals.json --no-llm --json --output reports/goals_multiple.json
```

`--rag` performs one local knowledge search after calculations. The default
makes one OpenAI request to explain the entire plan (not one call per goal).
`--no-llm` generates a local summary and skips OpenAI entirely. `.env` configures
OPENAI_API_KEY and OPENAI_MODEL without overriding environment variables. With
AI enabled, goal inputs, computed summaries, alternatives and selected PDF
excerpts are sent to OpenAI. Monthly transaction schedules are not sent.

## Input contract

See `src/data/sample_goals_plan.json` and `src/data/sample_multiple_goals.json`.

| Plan field | Meaning |
|---|---|
| as_of_date | Date of the input balances; YYYY-MM-DD |
| currency | One uppercase three-letter currency code for the entire plan |
| available_savings | Total savings actually available to assign to these goals |
| monthly_budget | Shared monthly contribution capacity |
| goals | One to twenty goals with unique IDs |

Each goal has an ID and name, target amount/date, target basis (`future_dollars`
or `today_dollars`), assigned starting balance, monthly contribution, annual
return and inflation assumptions, risk tolerance (`low`, `moderate`, `high`),
and optional annual fee, priority, deadline flexibility, and target flexibility.
Lower priority numbers have higher importance. Return and inflation assumptions
are required; they are not inferred from recent returns, RAG, or a risk label.
Annual fee defaults to zero when omitted and is always disclosed.

Input validation rejects nonfinite/negative money, money with more than two
decimal places, duplicate goal IDs, past target dates, horizons beyond 50 years,
over-assigned starting balances, and monthly commitments above the shared budget.
Rates and currencies are validated, with no currency conversion. Input dates
and declared availability do not establish account eligibility or withdrawal
access; users must supply funds actually available for these goals.

## Components and execution order

- `agents/goals/models.py`: validated goals, assumptions, and shared funding.
- `agents/goals/projection.py`: monthly Decimal calculations and required-saving solver.
- `agents/goals/planning.py`: budget-aware alternatives and educational risk guidance.
- `agents/goals/agent.py`: workflow and Markdown/JSON report.
- `rag/goals.py`: one post-calculation query across the local knowledge library.
- `agents/goals/explanation.py`: context, prompt, validation, and labeled fallback.
- `tools/openai_goals.py`: one strict-schema OpenAI request, 45-second timeout,
  no automatic retries. External API calls remain in tools.

The sequence is validate → calculate all goals and scenarios → optionally
retrieve once (20 candidates, best five usable chunks) → assemble report →
explain once. No initial routing LLM or second retrieval is involved.

## Projection algorithm

Monthly contributions occur at the end of full monthly anniversaries of the
plan date. Each anniversary is independently clamped to month-end: January 31
is followed by February 28/29 and March 31. Partial months before the target date
receive no growth or contribution. This convention is explicit, not a daily
interest calculation.

An effective annual return R becomes a monthly rate `(1 + R)^(1/12) - 1`.
An annual proportional fee F becomes monthly drag `1 - (1 - F)^(1/12)`.
Within each month:

1. Apply growth to the opening balance.
2. Charge the proportional fee on the post-growth balance.
3. Add the end-of-month contribution.
4. Use the result as next month's opening balance.

Cash flows are rounded to cents using ROUND_HALF_UP. Decimal arithmetic avoids
binary-float accumulation in the balances. Monthly rate conversion uses floating
point before Decimal conversion; these are projections, not brokerage accounting.

For today-dollar targets, inflate the amount to the actual deadline using the
supplied effective annual inflation and elapsed days / 365.25. Future-dollar
targets are already future amounts and are not inflated again. Taxes are omitted.

Required contribution uses a monotonic binary search for the smallest cent-sized
monthly deposit that reaches the target using the same monthly rounding model.
No monthly solution is invented when there are no deposit dates or the numerical
contribution bound is exceeded. A funded goal can require zero contributions.

## Alternatives and budget semantics

Every alternative is separate; none changes the baseline input:

- Required contribution at the original deadline, marked infeasible if it
  exceeds the budget remaining after other goals' current commitments.
- A later completion date with the original contribution, up to a 50-year
  search limit. Inflation follows the changed date for today-dollar targets.
- A smaller future-dollar target at the original deadline, only allowed when
  target flexibility was explicitly supplied.
- Additional assigned savings, using currently unassigned starting funds only.
- A priority-transfer option, using unassigned budget and lower-priority
  contributions. Donor reductions, projected balances and shortfalls are shown.

Date and target alternatives can be calculated for illustration while marked
not allowed by preferences. Budget feasibility is distinct from reaching the
target: a partial transfer can be affordable while leaving a shortfall. The
LLM's recommended scenario IDs must be present, allowed, and budget-feasible.
The report renders the calculated parameters and donor consequences directly
for each recommended option, even if the LLM prose omits those details.
Goals are not automatically sacrificed or reprioritized.

The plan-level required-contribution sum assumes fixed contributions beginning
now for all goals. It does not prove that dynamic reassignment after earlier
goals complete is impossible. Contributions stop at original goal deadlines;
freed money is not automatically moved to another goal.

## Risk appetite and scope

Educational policy `goals-risk-v1` uses a disclosed 36-month near-term screen.
A near-term goal emphasizes liquidity and capital preservation regardless of
high risk tolerance. For longer goals, low/moderate/high tolerance changes the
strategy considerations. The policy does not claim suitability or derive an
expected return from a risk label. It never raises the return assumption as a
remedy for inadequate funding.

A zero-gross-growth sensitivity uses the original fees and target assumptions.
It is not a worst-case loss bound and can be more optimistic than an originally
negative return assumption. Projected funding percentage measures balance /
target; it is not a probability of success. Full retirement withdrawals,
longevity, debt repayment, and tax optimization require separate models.

## Reports and explanation checks

Reports include the funding ledger; goal assumptions, monthly schedules, gap,
surplus and required contributions; budget capacity; independent scenarios;
risk guidance; source references; and limitations. JSON retains the full schedule.
Markdown shows the human-readable summaries. No plan is automatically executed.

The LLM receives summaries and valid field paths, not transaction schedules.
Validation checks goal coverage, goal IDs, referenced metric paths, source IDs,
numeric literals, and IDs of recommended scenarios. This does not prove semantic
correctness of prose or citation entailment. A failure/refusal/invalid response
retains the computed report and displays a clearly labeled local fallback.

## Hand-checkable example

The house sample has 15,000 assigned savings, a 60,000 future-dollar target,
36 monthly deposits of 900, zero growth, zero fees, and a 900 monthly budget.
It yields 47,400, a 12,600 shortfall, a required monthly contribution of 1,250,
and an extra budget requirement of 350. At the original contribution, the target
is reached after 50 monthly deposits. The multi-goal sample demonstrates shared
resources, inflation, fees, fixed preferences, and explicitly shown donor effects.
