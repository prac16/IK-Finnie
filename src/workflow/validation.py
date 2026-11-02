"""Resolve authoritative inputs and validate before any specialist/API call."""
from copy import deepcopy
from pydantic import ValidationError
from agents.goals.models import GoalPlan
from rag.categories import CATEGORIES
from workflow.schemas import (TurnRequest, RouteDecision, MarketArguments, GoalsArguments,
                              PortfolioArguments, ChatArguments)


def _issues(exc: ValidationError) -> list[str]:
    return [('.'.join(map(str, e['loc'])) or 'inputs') + ': ' + e['msg'] for e in exc.errors(include_url=False, include_input=False)][:20]


def validate_request(request: TurnRequest, decision: RouteDecision, context: dict, store, thread_id: str) -> dict:
    if decision.action in {'clarify', 'unsupported'} or decision.agent is None:
        return {'ready': False, 'unsupported': decision.action == 'unsupported', 'missing': ['intent'],
                'question': decision.question or 'Which analysis would you like: chat, portfolio, market, or goals?'}
    agent, args = decision.agent, decision.arguments
    try:
        if decision.action == 'explain':
            if agent == 'chat':
                raise ValueError('Select a portfolio, market, or goals report to explain')
            if args:
                raise ValueError('Input changes require a new analysis rather than an explanation of an old report')
            report_id = context.get('reports', {}).get(agent)
            if not report_id:
                raise ValueError('No saved report is selected; run an analysis first')
            store.metadata(thread_id, report_id)
            return {'ready': True, 'arguments': {}, 'report_id': report_id, 'action': 'explain'}
        if agent == 'market':
            inputs = MarketArguments.model_validate(context.get('market', {}) | args)
            if inputs.benchmark == inputs.symbol:
                raise ValueError('Benchmark must differ from the analyzed ticker')
            return {'ready': True, 'arguments': inputs.model_dump(), 'action': 'analyze'}
        if agent == 'portfolio':
            inputs = PortfolioArguments.model_validate(args)
            selected = inputs.portfolio_id or context.get('portfolio_id')
            options = [m for m in store.list(thread_id) if m['kind'] == 'portfolio']
            if not selected and len(options) == 1:
                selected = options[0]['id']
            if not selected:
                return {'ready': False, 'missing': ['portfolio_id'],
                        'question': 'Upload holdings as CSV/XLSX, or select a portfolio_id from the listed portfolios.', 'options': options}
            store.get(thread_id, selected, kind='portfolio')
            return {'ready': True, 'arguments': {'portfolio_id': selected}, 'input_id': selected, 'action': 'analyze'}
        if agent == 'chat':
            inputs = ChatArguments.model_validate(args)
            if inputs.category is not None and inputs.category not in CATEGORIES:
                raise ValueError('Choose a valid knowledge category: ' + ', '.join(CATEGORIES))
            return {'ready': True, 'arguments': {'question': inputs.question or request.message, 'category': inputs.category}, 'action': 'analyze'}
        inputs = GoalsArguments.model_validate(args)
        plan_id = inputs.plan_id or context.get('plan_id')
        if plan_id is None and inputs.plan is None:
            options = [m for m in store.list(thread_id) if m['kind'] == 'goal_plan']
            if len(options) == 1:
                plan_id = options[0]['id']
            elif len(options) > 1:
                return {'ready': False, 'missing': ['plan_id'], 'options': options,
                        'question': 'Select a plan_id from the uploaded goal plans.'}
        plan = deepcopy(inputs.plan) if inputs.plan is not None else (store.get(thread_id, plan_id, kind='goal_plan') if plan_id else {})
        plan.update(inputs.plan_updates)
        if inputs.goal_updates:
            goals = plan.get('goals', [])
            selected = inputs.goal_id or (goals[0].get('id') if len(goals) == 1 else None)
            matches = [g for g in goals if g.get('id') == selected]
            if len(matches) != 1:
                raise ValueError('Select a goal_id before editing a plan with multiple goals')
            matches[0].update(inputs.goal_updates)
        try:
            parsed = GoalPlan.model_validate(plan)
        except ValidationError as exc:
            # Retain the draft for clarification, without committing it to active context.
            return {'ready': False, 'missing': _issues(exc), 'draft_plan': plan,
                    'question': 'Please supply or correct these goal-plan fields: ' + '; '.join(_issues(exc))}
        return {'ready': True, 'arguments': {'plan': parsed.model_dump(mode='json')}, 'parent_id': plan_id, 'action': 'analyze'}
    except ValidationError as exc:
        issues = _issues(exc)
        return {'ready': False, 'missing': issues, 'question': 'Please supply or correct: ' + '; '.join(issues)}
    except (ValueError, OSError) as exc:
        # These errors come from our local validation, never provider exceptions.
        return {'ready': False, 'missing': ['inputs'], 'question': str(exc) if isinstance(exc, ValueError) else 'Input artifact unavailable; upload it again.'}
