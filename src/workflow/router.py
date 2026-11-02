"""Narrow rule fast paths, followed by at most one structured routing call."""
from __future__ import annotations

import re
from workflow.schemas import RouteDecision, TurnRequest
from tools.openai_router import RoutingDraft


def extract_market_fields(message: str) -> dict:
    fields = {}
    sessions = re.search(r'\b(\d+)\s*(?:trading\s+)?sessions?\b', message, re.I)
    if sessions:
        fields['sessions'] = int(sessions[1])
    benchmark = re.search(r'(?i:\b(?:against|versus|vs\.?|benchmark(?:ed)?(?:\s+to)?)\s+)\$?([A-Z][A-Z0-9.\-]{0,9})\b', message)
    if benchmark:
        fields['benchmark'] = benchmark[1]
    ticker = re.search(r'(?i:\banaly[sz]e\s+|\bwhat about\s+|\bfor\s+|\bticker\s+|\$)([A-Z][A-Z0-9.\-]{0,9})\b', message)
    if ticker and ticker[1] not in {'I', 'API', 'CSV', 'USD', 'ETF'}:
        fields['symbol'] = ticker[1]
    if re.fullmatch(r'\$?[A-Z][A-Z0-9.\-]{0,9}', message.strip()):
        fields['symbol'] = message.strip().lstrip('$')
    if re.search(r'\bunadjusted\b|\braw (?:close|prices?)\b', message, re.I):
        fields['adjusted'] = False
    elif re.search(r'\badjusted\b', message, re.I):
        fields['adjusted'] = True
    return fields


def route_request(request: TurnRequest, context: dict, artifacts: list[dict], messages: list,
                  *, generator=None) -> RouteDecision:
    text, lower = request.message, request.message.casefold()
    market = extract_market_fields(text)
    concept = bool(re.match(r'^(?:what (?:is|are)|define|explain the (?:concept|meaning) of)\b', lower))
    followup = bool(re.match(r'^(?:explain (?:that|this|the report)|why (?:is|was) (?:that|this)|what does (?:that|this))\b', lower))
    contribution = re.search(r'\b(?:contribute|monthly contribution(?:\s+to|\s+of)?)\s*\$?([\d,]+(?:\.\d{1,2})?)\b', lower)
    increment = bool(re.search(r'\b(?:more|extra|increase|decrease|less)\b', lower))
    goal_args = {}
    if contribution and not increment:
        goal_args['goal_updates'] = {'monthly_contribution': float(contribution[1].replace(',', ''))}
    budget = re.search(r'\bmonthly budget\s+(?:to\s+|of\s+)?\$?([\d,]+(?:\.\d{1,2})?)\b', lower)
    if budget:
        goal_args['plan_updates'] = {'monthly_budget': float(budget[1].replace(',', ''))}
    # Explicit route is authoritative, while narrowly parsed fields are still useful.
    if request.route:
        if request.route == 'goals' and contribution and increment and not request.arguments.get('goal_updates'):
            return RouteDecision(agent='goals', action='clarify', question='Supply the absolute monthly contribution and any explicit shared-budget change.')
        args = market if request.route == 'market' else goal_args if request.route == 'goals' else {}
        return RouteDecision(agent=request.route, arguments=args | request.arguments, method='explicit')
    if re.search(r'\b(?:news|headlines)\b', lower) and not concept:
        return RouteDecision(action='unsupported', question='News synthesis is not available yet. I can analyze historical market metrics instead.')
    if re.search(r'\b(?:buy|sell|place|execute)\b.*\b(?:order|trade)\b', lower):
        return RouteDecision(action='unsupported', question='Trade execution is not supported.')
    if re.search(r'\b(?:then|and then)\b', lower) and re.search(r'\bportfolio\b', lower) and re.search(r'\bgoal', lower):
        return RouteDecision(action='clarify', question='Please choose portfolio analysis or goals planning first. Combined workflows are a later phase.')
    if 'plan' in request.arguments or 'plan_id' in request.arguments or 'goal_updates' in request.arguments or 'plan_updates' in request.arguments:
        return RouteDecision(agent='goals', arguments=request.arguments)
    if 'portfolio_id' in request.arguments:
        return RouteDecision(agent='portfolio', arguments=request.arguments)
    if 'symbol' in request.arguments:
        return RouteDecision(agent='market', arguments=market | request.arguments)
    if concept:
        return RouteDecision(agent='chat', arguments=request.arguments)
    if followup and context.get('last_agent') and context.get('reports', {}).get(context['last_agent']):
        if context['last_agent'] != 'chat':
            return RouteDecision(agent=context['last_agent'], action='explain')
    if re.search(r'\b(?:portfolio|holdings|position|concentrat\w*)\b', lower):
        return RouteDecision(agent='portfolio', arguments=request.arguments)
    if contribution and increment:
        return RouteDecision(agent='goals', action='clarify', question='What absolute monthly contribution should I use? Please also state any change to the shared monthly budget.')
    if goal_args or re.search(r'\b(?:goals?|retirement|house deposit|monthly contribution)\b', lower):
        return RouteDecision(agent='goals', arguments=goal_args | request.arguments)
    if re.search(r'\b(?:analy[sz]e|market|stock|ticker)\b', lower) and ('symbol' in market or 'market' in lower or 'ticker' in lower):
        return RouteDecision(agent='market', arguments=market | request.arguments)
    if re.match(r'^what about\b', lower) and 'symbol' in market and context.get('last_agent') == 'market':
        return RouteDecision(agent='market', arguments=market | request.arguments)
    if not request.use_llm:
        return RouteDecision(action='clarify', question='Choose chat, portfolio, market, or goals, and supply structured inputs. Natural-language routing is disabled.')
    try:
        if generator is None:
            from tools.openai_router import OpenAIRouter
            generator = OpenAIRouter()
        # Only concise selected facts; no full reports, uploads, or schedules.
        selected = {k: context[k] for k in ('last_agent', 'market', 'portfolio_id', 'plan_id', 'goal_ids') if k in context}
        draft = RoutingDraft.model_validate(generator.route({
            'message': text, 'active_context': selected,
            'artifacts': [{k: a[k] for k in ('id', 'kind', 'name')} for a in artifacts[-20:]],
            'recent_messages': [{'role': m.type, 'content': str(m.content)[:1000]} for m in messages[-6:]],
        }))
        args = {}
        if draft.agent == 'market':
            args = {k: getattr(draft, k) for k in ('symbol', 'sessions', 'benchmark', 'adjusted') if getattr(draft, k) is not None}
        if draft.agent == 'goals':
            if draft.goal_id is not None:
                args['goal_id'] = draft.goal_id
            if draft.monthly_contribution is not None:
                args['goal_updates'] = {'monthly_contribution': draft.monthly_contribution}
            if draft.monthly_budget is not None:
                args['plan_updates'] = {'monthly_budget': draft.monthly_budget}
        action = draft.action
        if action == 'analyze' and draft.agent is None:
            action = 'clarify'
        return RouteDecision(agent=draft.agent, action=action, arguments=args | request.arguments,
                             question=draft.question, method='llm', router_calls=1)
    except Exception:
        return RouteDecision(action='clarify', method='llm_unavailable', router_calls=1,
                             question='I could not determine the request. Choose chat, portfolio, market, or goals and provide the relevant inputs.')
