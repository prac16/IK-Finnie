"""Goals workflow: validate → project/scenarios → one optional RAG → one LLM."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError
from agents.goals.models import GoalPlan
from agents.goals.planning import GoalResult, analyze_goal
from agents.goals.projection import cash
from agents.goals.explanation import GoalsExplanation, explain_goals_report
from rag.goals import retrieve_goals_knowledge
from rag.ingest import DEFAULT_PDF_DIR
from rag.retrieval import QueryCollection, RetrievalResult
from rag.store import local_collection
from tools.openai_goals import GoalsExplanationGenerator


@dataclass(frozen=True)
class GoalsReport:
    plan: dict
    funding: dict
    results: tuple[GoalResult, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]
    knowledge: RetrievalResult | None = None
    explanation: GoalsExplanation | None = None
    policy_version: str = 'goals-risk-v1'

    def to_dict(self): return asdict(self)

    def to_markdown(self):
        lines=['# Financial goals plan',f"As of {self.plan['as_of_date']}; currency {self.plan['currency']}."]
        if self.explanation: lines.append(self.explanation.to_markdown())
        lines.extend(['## Shared funding', '\n'.join([
            '| Measure | Amount |','|---|---:|',
            f"| Available savings | {self.plan['available_savings']:,.2f} |",
            f"| Assigned starting balances | {self.funding['committed_savings']:,.2f} |",
            f"| Unassigned savings | {self.funding['unassigned_savings']:,.2f} |",
            f"| Monthly budget | {self.plan['monthly_budget']:,.2f} |",
            f"| Assigned monthly contributions | {self.funding['committed_monthly']:,.2f} |",
            f"| Unassigned monthly budget | {self.funding['unassigned_monthly']:,.2f} |",
        ])])
        for result in self.results:
            goal,p=result.goal,result.projection
            lines.extend([f"## {goal['name']} ({goal['id']})",f"Status: {p.status.replace('_',' ')}; deadline {p.deadline}; priority {goal['priority']} (1 is highest).",
                f"Assumptions: annual return {goal['annual_return_pct']:.2f}%, inflation {goal['annual_inflation_pct']:.2f}%, fee {goal['annual_fee_pct']:.2f}%; target basis {goal['target_basis']}.",
                '\n'.join(['| Measure | Value |','|---|---:|',
                    f'| Monthly contributions before deadline | {p.contribution_months} |',
                    f'| Future target | {p.future_target:,.2f} |',f'| Projected balance | {p.ending_balance:,.2f} |',
                    f'| Projected shortfall | {p.shortfall:,.2f} |',f'| Projected surplus | {p.surplus:,.2f} |',
                    f'| Projected funding (not probability) | {p.projected_funding_pct:.2f}% |',
                    f'| Required monthly contribution | {result.required_monthly:,.2f} |' if result.required_monthly is not None else '| Required monthly contribution | Unavailable: no monthly solution within bounds |',
                    f'| Monthly capacity with other goals unchanged | {result.available_monthly_for_this_goal:,.2f} |',
                    f'| Extra monthly budget needed | {result.additional_budget_needed:,.2f} |' if result.additional_budget_needed is not None else '| Extra monthly budget needed | Unavailable |',
                    f'| Zero gross-growth projected balance (same fees) | {result.zero_growth_projection.ending_balance:,.2f} |',
                ]), '### Risk and horizon guidance', '\n'.join('- '+s for s in result.risk_guidance)])
            if result.scenarios:
                lines.append('### Independent scenarios — not selected changes')
            for scenario in result.scenarios:
                lines.extend([f"**{scenario.kind.replace('_',' ')}** — allowed by preferences: {'yes' if scenario.allowed_by_preferences else 'no'}; budget feasible: {'yes' if scenario.feasible_with_budget else 'no'}.",scenario.description])
                details=[]
                for label,value in (('Monthly contribution',scenario.monthly_contribution),('Additional assigned savings',scenario.additional_starting_balance),('Target alternative (future dollars)',scenario.target_amount),('Projected balance',scenario.projected_balance),('Shortfall',scenario.shortfall)):
                    if value is not None: details.append(f'{label}: {value:,.2f}')
                if scenario.deadline: details.append('Deadline: '+scenario.deadline)
                if details: lines.append('; '.join(details)+'.')
                for affected in scenario.affected_goals:
                    lines.append(f"- Donor {affected['goal_id']}: monthly contribution becomes {affected['new_monthly_contribution']:,.2f}; resulting shortfall {affected['shortfall']:,.2f}.")
        if self.knowledge:
            lines.append('## Financial knowledge references')
            for index,chunk in enumerate(self.knowledge.chunks,1):
                url=quote(str(DEFAULT_PDF_DIR/chunk.source),safe='/')
                lines.extend([f'[{index}] [{chunk.source}, p. {chunk.page}]({url}#page={chunk.page})','> '+chunk.text])
            lines.append(f'Retrieval: {self.knowledge.metrics.retrieval_ms:.1f} ms; {len(self.knowledge.chunks)} selected chunks.')
        lines.extend(['## Model limitations','\n'.join('- '+s for s in self.limitations)])
        if self.warnings: lines.extend(['## Warnings','\n'.join('- '+s for s in self.warnings)])
        return '\n\n'.join(lines)


class GoalsPlanningAgent:
    def __init__(self, *, collection: QueryCollection | None = None, use_llm=True,
                 generator: GoalsExplanationGenerator | None = None):
        self.collection,self.use_llm,self.generator=collection,use_llm,generator

    def plan(self, inputs: GoalPlan | dict) -> GoalsReport:
        # Revalidate even a model supplied via model_copy/model_construct.
        plan=GoalPlan.model_validate(inputs.model_dump() if isinstance(inputs,GoalPlan) else inputs)
        results=tuple(analyze_goal(goal,plan) for goal in plan.goals)
        committed_savings=sum(cash(g.starting_balance) for g in plan.goals)
        committed_monthly=sum(cash(g.monthly_contribution) for g in plan.goals)
        requireds=[result.required_monthly for result in results]
        total_required=sum(cash(value) for value in requireds) if all(v is not None for v in requireds) else None
        funding={'committed_savings':float(committed_savings),'unassigned_savings':float(cash(plan.available_savings)-committed_savings),
                 'committed_monthly':float(committed_monthly),'unassigned_monthly':float(cash(plan.monthly_budget)-committed_monthly),
                 'required_monthly_total_fixed_contributions':float(total_required) if total_required is not None else None,
                 'fixed_contribution_budget_gap':float(max(cash(0),total_required-cash(plan.monthly_budget))) if total_required is not None else None}
        warnings=[]
        if funding['fixed_contribution_budget_gap']:
            warnings.append('Required fixed monthly contributions for all goals exceed the shared budget under the supplied assumptions; alternative scenarios are independent.')
        if any(r.required_monthly is None for r in results):
            warnings.append('At least one goal has no monthly solution before its deadline or within the contribution bound; no amount is invented.')
        limitations=(
            'Educational projections conditional on user-supplied returns and inflation; not forecasts, guarantees, or probabilities of success.',
            'Monthly contributions occur at the end of each full monthly anniversary, clamped to month-end; partial months have no growth or deposit.',
            'Growth is applied before the monthly fee and end-of-month contribution; amounts are rounded to cents each month.',
            'Annual growth is an effective rate; annual fees are modeled as a proportional drag converted to monthly. Taxes are not modeled.',
            'Inflation applies only to targets expressed in today dollars, using actual days / 365.25; future-dollar targets are not inflated again.',
            'Contributions stop at each goal deadline. Funds freed by completed goals are not automatically reassigned.',
            'Available savings and budget are user-declared; account restrictions, withdrawal penalties and access dates are not inferred. Use only funds available for these goals.',
            'Risk screens use disclosed educational rules. No asset allocation, holdings selection, trading, market/news calls, or Monte Carlo simulation is performed.',
            'Scenarios keep other commitments unchanged unless donor effects are shown. Alternatives must not be combined without recalculation.',
        )
        knowledge=None
        if self.collection is not None:
            try:
                knowledge=retrieve_goals_knowledge(self.collection,results)
                if not knowledge.chunks: warnings.append('No usable financial knowledge excerpts retrieved.')
            except Exception:
                warnings.append('Financial knowledge retrieval unavailable; calculated plan remains available.')
        report=GoalsReport(plan.model_dump(mode='json'),funding,results,limitations,tuple(warnings),knowledge)
        return replace(report,explanation=explain_goals_report(report,use_llm=self.use_llm,generator=self.generator))


def main():
    parser=argparse.ArgumentParser(description='Educational multi-goal funding planner with explicit assumptions.')
    parser.add_argument('plan_file',type=Path,help='JSON goals and shared resources')
    parser.add_argument('--rag',action='store_true',help='Retrieve goal-related knowledge once after projection')
    parser.add_argument('--no-llm',action='store_true',help='Use a local summary without sending the plan to OpenAI')
    parser.add_argument('--json',action='store_true',help='Export structured results including monthly schedules')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    collection,warning=None,None
    if args.rag:
        try: collection=local_collection()
        except Exception: warning='Local RAG index unavailable; no knowledge excerpts attached.'
    try:
        if args.plan_file.stat().st_size > 1024*1024: raise ValueError('Plan input exceeds 1 MB')
        inputs=json.loads(args.plan_file.read_text())
        report=GoalsPlanningAgent(collection=collection,use_llm=not args.no_llm).plan(inputs)
    except (OSError,ValueError,ValidationError) as exc:
        parser.error(str(exc))
    if warning: report=replace(report,warnings=(*report.warnings,warning))
    text=json.dumps(report.to_dict(),indent=2,allow_nan=False) if args.json else report.to_markdown()
    if args.output: args.output.write_text(text)
    else: print(text)


if __name__=='__main__': main()
