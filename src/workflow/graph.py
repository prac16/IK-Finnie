"""LangGraph coordination: route → validate → specialist, or interrupt."""
from __future__ import annotations

from time import perf_counter
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from workflow.router import route_request
from workflow.schemas import AgentResult, RouteDecision, TurnRequest
from workflow.state import WorkflowState
from workflow.validation import validate_request


def build_graph(store, adapters, *, router_generator=None, checkpointer=None, on_event=None):
    def event(name, request_id):
        if on_event:
            try:
                on_event({'stage': name, 'request_id': request_id})
            except Exception:
                pass  # A progress display must not prevent analysis.

    def normalize(state, config):
        request = TurnRequest.model_validate(state['request'])
        thread = config['configurable']['thread_id']
        registry = store.list(thread)
        old_ids = {m['id'] for m in state.get('artifact_registry', [])}
        context = dict(state.get('active_context', {}))
        for kind, key in [('portfolio', 'portfolio_id'), ('goal_plan', 'plan_id')]:
            uploaded = [m for m in registry if m['kind'] == kind and m['id'] not in old_ids]
            if uploaded:
                context.pop(key, None)
                if kind == 'goal_plan':
                    context.pop('goal_ids', None)
                if len(uploaded) == 1:
                    context[key] = uploaded[0]['id']
                    if kind == 'goal_plan':
                        context['goal_ids'] = [g['id'] for g in store.get(thread, uploaded[0]['id'], kind=kind)['goals']]
        return {'request': request.model_dump(mode='json'), 'messages': [HumanMessage(content=request.message)],
                'decision': {}, 'validated': {}, 'result': None, 'pending_request': None, 'final_response': '',
                'artifact_registry': registry, 'active_context': context,
                'metrics': {'router_calls': 0, 'route_ms': 0.0, 'validation_ms': 0.0, 'execution_ms': 0.0}}

    def route(state):
        request = TurnRequest.model_validate(state['request'])
        event('routing', request.request_id)
        started = perf_counter()
        decision = route_request(request, state['active_context'], state['artifact_registry'], state.get('messages', []),
                                 generator=router_generator)
        metrics = dict(state['metrics'])
        metrics['router_calls'] += decision.router_calls
        metrics['route_ms'] += round((perf_counter() - started) * 1000, 2)
        return {'decision': decision.model_dump(mode='json'), 'metrics': metrics, 'pending_request': None}

    def validate(state, config):
        started = perf_counter()
        checked = validate_request(TurnRequest.model_validate(state['request']), RouteDecision.model_validate(state['decision']),
                                   state['active_context'], store, config['configurable']['thread_id'])
        if not checked.get('ready') and not checked.get('unsupported'):
            event('needs_input', state['request']['request_id'])
        metrics = dict(state['metrics'])
        metrics['validation_ms'] += round((perf_counter() - started) * 1000, 2)
        return {'validated': checked, 'pending_request': None if checked.get('ready') or checked.get('unsupported') else checked,
                'metrics': metrics}

    def clarify(state):
        # No external work before this call: this node restarts on resume.
        reply = interrupt({'question': state['validated']['question'], 'missing': state['validated'].get('missing', []),
                           'options': state['validated'].get('options', [])})
        request = TurnRequest.model_validate(reply)
        return {'request': request.model_dump(mode='json'), 'messages': [HumanMessage(content=request.message)],
                'pending_request': None, 'final_response': '', 'result': None}

    def execute(state, config):
        request = TurnRequest.model_validate(state['request'])
        agent = state['decision']['agent']
        thread = config['configurable']['thread_id']
        event('explaining_saved_report' if state['validated']['action'] == 'explain' else 'analyzing', request.request_id)
        started = perf_counter()
        try:
            result = adapters.execute(agent, state['validated'], request, thread)
            result = AgentResult.model_validate(result)
        except Exception:
            # Only ordinary specialist errors reach here; interrupts live in their own node.
            result = AgentResult(agent=agent, request_id=request.request_id, status='error',
                                 markdown='The analysis could not be completed. Check your inputs and provider configuration, then try again.')
        context = dict(state['active_context'])
        if result.status != 'error':
            context['last_agent'] = agent
            reports = dict(context.get('reports', {}))
            if result.report_id:
                reports[agent] = result.report_id
            else:
                reports.pop(agent, None)
            context['reports'] = reports
            if state['validated']['action'] != 'explain':
                if agent == 'market':
                    context['market'] = state['validated']['arguments']
                elif agent == 'portfolio':
                    context['portfolio_id'] = result.input_id
                elif agent == 'goals':
                    context['plan_id'] = result.input_id
                    context['goal_ids'] = [g['id'] for g in state['validated']['arguments']['plan']['goals']]
        metrics = dict(state['metrics'])
        metrics['execution_ms'] = round((perf_counter() - started) * 1000, 2)
        return {'result': result.model_dump(mode='json'), 'active_context': context, 'metrics': metrics,
                'artifact_registry': store.list(thread)}

    def finish(state):
        request = TurnRequest.model_validate(state['request'])
        text = state['result']['markdown'] if state.get('result') else state['validated'].get('question', 'Request unavailable.')
        event('complete', request.request_id)
        return {'final_response': text, 'pending_request': None, 'messages': [AIMessage(content=text)]}

    builder = StateGraph(WorkflowState)
    for name, node in [('normalize', normalize), ('route', route), ('validate', validate), ('clarify', clarify),
                       ('execute', execute), ('finish', finish)]:
        builder.add_node(name, node)
    builder.add_edge(START, 'normalize')
    builder.add_edge('normalize', 'route')
    builder.add_edge('route', 'validate')
    builder.add_conditional_edges('validate', lambda s: 'finish' if s['validated'].get('unsupported') else
                                  ('execute' if s['validated'].get('ready') else 'clarify'),
                                  {'finish': 'finish', 'execute': 'execute', 'clarify': 'clarify'})
    builder.add_edge('clarify', 'route')
    builder.add_edge('execute', 'finish')
    builder.add_edge('finish', END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
