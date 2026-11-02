"""Budget-aware alternatives and transparent educational risk policy."""
from __future__ import annotations

from dataclasses import dataclass

from agents.goals.models import Goal, GoalPlan
from agents.goals.projection import Projection, add_months, cash, months_available, project, required_monthly


@dataclass(frozen=True)
class GoalScenario:
    kind: str
    allowed_by_preferences: bool
    feasible_with_budget: bool
    description: str
    monthly_contribution: float | None = None
    target_amount: float | None = None
    deadline: str | None = None
    projected_balance: float | None = None
    shortfall: float | None = None
    affected_goals: tuple[dict, ...] = ()
    additional_starting_balance: float | None = None


@dataclass(frozen=True)
class GoalResult:
    goal: dict
    projection: Projection
    zero_growth_projection: Projection
    required_monthly: float | None
    available_monthly_for_this_goal: float
    additional_monthly_needed: float | None
    additional_budget_needed: float | None
    risk_guidance: tuple[str, ...]
    scenarios: tuple[GoalScenario, ...]


def risk_guidance(goal: Goal, months: int) -> tuple[str, ...]:
    guidance = [f'Declared risk tolerance: {goal.risk_tolerance}. The growth rate is a user assumption, not an expected return derived from this risk label.']
    if months <= 36:
        guidance.append('Near-term horizon (36 months or less): evaluate liquidity and capital preservation first; high willingness to accept losses does not remove deadline risk.')
    elif goal.risk_tolerance == 'low':
        guidance.append('Low tolerance: evaluate capital-preservation-oriented approaches and the zero-growth scenario before accepting reliance on investment gains.')
    elif goal.risk_tolerance == 'moderate':
        guidance.append('Moderate tolerance and a longer horizon: evaluate a diversified mix and how losses could affect the target; no asset percentages are prescribed.')
    else:
        guidance.append('High tolerance and a longer horizon: diversified growth exposure is an option to evaluate, with possible losses; higher tolerance does not guarantee the target.')
    if not goal.deadline_flexible:
        guidance.append('The deadline is fixed: a delay is an illustration only, not an acceptable change under the supplied preferences.')
    if not goal.target_flexible:
        guidance.append('The target is fixed: lowering it is not an acceptable change under the supplied preferences.')
    guidance.append('Educational policy goals-risk-v1; horizon screens are not a suitability assessment or a loss-probability model.')
    return tuple(guidance)


def delayed_completion(goal: Goal, plan: GoalPlan) -> Projection | None:
    """Search later monthly dates, stopping after 50 years. Includes inflation at each date."""
    first = months_available(plan.as_of_date, goal.target_date) + 1
    # Simulating one growing prefix is sufficient; avoid O(months²) searches.
    end = add_months(plan.as_of_date, 600)
    trajectory = project(goal, plan.as_of_date, deadline=end)
    from agents.goals.projection import target_at
    for index, row in enumerate(trajectory.schedule, start=1):
        candidate_date = add_months(plan.as_of_date, index)
        if index >= first and candidate_date > goal.target_date and cash(row.closing_balance) >= target_at(goal, plan.as_of_date, candidate_date):
            return project(goal, plan.as_of_date, deadline=candidate_date, include_schedule=False)
    return None


def analyze_goal(goal: Goal, plan: GoalPlan) -> GoalResult:
    baseline = project(goal, plan.as_of_date)
    zero = project(goal, plan.as_of_date, annual_return_pct=0, include_schedule=False)
    required = required_monthly(goal, plan.as_of_date)
    other_commitments = sum(cash(g.monthly_contribution) for g in plan.goals if g.id != goal.id)
    capacity = cash(plan.monthly_budget) - other_commitments
    scenarios = []
    if baseline.shortfall > 0:
        increased = project(goal, plan.as_of_date, monthly_contribution=required, include_schedule=False) if required is not None else None
        scenarios.append(GoalScenario('required_contribution', True, required is not None and cash(required) <= capacity,
            'Required saving with other goals unchanged; amounts above the stated capacity require a budget change, not an assumed extra income source.',
            required, deadline=goal.target_date.isoformat(), projected_balance=increased.ending_balance if increased else None,
            shortfall=increased.shortfall if increased else None))
        delayed = delayed_completion(goal, plan)
        scenarios.append(GoalScenario('later_deadline', goal.deadline_flexible, goal.deadline_flexible and delayed is not None,
            'Keep the contribution unchanged; future targets in today dollars continue to inflate. No completion date is claimed if the target is not reached within 50 years.',
            goal.monthly_contribution, deadline=delayed.deadline if delayed else None,
            projected_balance=delayed.ending_balance if delayed else None, shortfall=delayed.shortfall if delayed else None))
        scenarios.append(GoalScenario('smaller_target', goal.target_flexible, goal.target_flexible,
            'Illustrative smaller target in future dollars at the original deadline; applies only if the target is flexible.',
            goal.monthly_contribution, target_amount=baseline.ending_balance, deadline=goal.target_date.isoformat(),
            projected_balance=baseline.ending_balance, shortfall=0.0))
        unassigned = cash(plan.available_savings)-sum(cash(g.starting_balance) for g in plan.goals)
        if unassigned > 0:
            def with_extra(cents):
                revised_goal=goal.model_copy(update={'starting_balance':float(cash(goal.starting_balance)+cents/100)})
                return project(revised_goal,plan.as_of_date,include_schedule=False)
            from decimal import Decimal
            low,high=0,int(unassigned*100)
            if with_extra(Decimal(high)).shortfall == 0:
                while low+1 < high:
                    mid=(low+high)//2
                    if with_extra(Decimal(mid)).shortfall == 0: high=mid
                    else: low=mid
            revised=with_extra(Decimal(high))
            scenarios.append(GoalScenario('additional_lump_sum',True,True,
                'Use only currently unassigned savings; the extra amount cannot also fund another goal. Monthly commitments are unchanged.',
                goal.monthly_contribution,deadline=goal.target_date.isoformat(),projected_balance=revised.ending_balance,
                shortfall=revised.shortfall,additional_starting_balance=float(Decimal(high)/100)))
        # One transparent, non-mutating priority-transfer alternative. Show donor harm.
        if required is not None and required > goal.monthly_contribution:
            needed = max(cash(0), cash(required)-cash(goal.monthly_contribution))
            available_extra = max(cash(0), capacity-cash(goal.monthly_contribution))
            remaining = max(cash(0), needed-available_extra)
            effects = []
            transferred = cash(0)
            donors = sorted((g for g in plan.goals if g.priority > goal.priority and g.monthly_contribution > 0),
                            key=lambda g: (-g.priority,g.id))
            for donor in donors:
                take = min(cash(donor.monthly_contribution), remaining)
                if take <= 0: break
                reduced = cash(donor.monthly_contribution)-take
                revised = project(donor, plan.as_of_date, monthly_contribution=float(reduced), include_schedule=False)
                effects.append({'goal_id': donor.id, 'reduction': float(take), 'new_monthly_contribution': float(reduced),
                                'projected_balance': revised.ending_balance, 'shortfall': revised.shortfall})
                transferred += take
                remaining -= take
            if effects:
                recipient_monthly = cash(goal.monthly_contribution)+min(needed,available_extra+transferred)
                recipient = project(goal, plan.as_of_date, monthly_contribution=float(recipient_monthly), include_schedule=False)
                scenarios.append(GoalScenario('priority_reallocation', True, True,
                    'Independent budget-feasible option using unassigned budget and lower-priority contributions. It changes the donor goals; it does not automatically select this plan or guarantee the recipient is funded.',
                    float(recipient_monthly), deadline=goal.target_date.isoformat(), projected_balance=recipient.ending_balance,
                    shortfall=recipient.shortfall, affected_goals=tuple(effects)))
    return GoalResult(goal.model_dump(mode='json'), baseline, zero, required, float(capacity),
                      max(0, float(cash(required)-cash(goal.monthly_contribution))) if required is not None else None,
                      max(0, float(cash(required)-capacity)) if required is not None else None,
                      risk_guidance(goal, baseline.contribution_months), tuple(scenarios))
