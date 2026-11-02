"""Process-local conversation API shared by CLI and a future UI."""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from langgraph.types import Command
from workflow.adapters import AgentAdapters
from workflow.artifacts import ArtifactStore
from workflow.graph import build_graph
from workflow.schemas import AgentResult, TurnRequest, TurnResponse


class WorkflowService:
    def __init__(self, *, store=None, adapters=None, router_generator=None, checkpointer=None, on_event=None):
        self.store = store if store is not None else ArtifactStore()
        self.adapters = adapters if adapters is not None else AgentAdapters(self.store)
        self.graph = build_graph(self.store, self.adapters, router_generator=router_generator,
                                 checkpointer=checkpointer, on_event=on_event)
        # A local service serializes mutations, including register/submit/resume.
        self._lock = RLock()

    @staticmethod
    def _config(thread_id):
        if not isinstance(thread_id, str) or not thread_id.strip() or len(thread_id) > 128:
            raise ValueError('thread_id must be a nonblank string of at most 128 characters')
        return {'configurable': {'thread_id': thread_id}, 'recursion_limit': 16, 'callbacks': []}

    def get_state(self, thread_id):
        with self._lock:
            return deepcopy(self.graph.get_state(self._config(thread_id)).values)

    def add_holdings(self, thread_id, content: bytes, filename: str):
        with self._lock:
            self._config(thread_id)
            return self.store.add_holdings(thread_id, content, filename)

    def add_plan(self, thread_id, plan: dict):
        with self._lock:
            self._config(thread_id)
            return self.store.add_plan(thread_id, plan)

    def get_report(self, thread_id, report_id):
        with self._lock:
            self._config(thread_id)
            return self.store.get(thread_id, report_id, kind='report')

    def submit(self, thread_id, request: TurnRequest | dict | str):
        with self._lock:
            config = self._config(thread_id)
            if self.graph.get_state(config).next:
                raise ValueError('Clarification is pending; resume it or cancel before submitting a new request')
            if isinstance(request, str):
                request = TurnRequest(message=request)
            request = TurnRequest.model_validate(request)
            self.graph.invoke({'request': request.model_dump(mode='json')}, config)
            return self._response(thread_id)

    def resume(self, thread_id, reply: TurnRequest | dict | str):
        with self._lock:
            config = self._config(thread_id)
            state = self.graph.get_state(config)
            if not state.next or not state.values.get('pending_request'):
                raise ValueError('No clarification is pending')
            original = TurnRequest.model_validate(state.values['request'])
            decision = state.values['decision']
            checked = state.values['validated']
            args = deepcopy(decision.get('arguments', {}))
            if 'draft_plan' in checked:
                args = {'plan': checked['draft_plan'], **{k: v for k, v in args.items() if k in {'plan_id', 'goal_id'}}}
            if isinstance(reply, str):
                reply = {'message': reply}
            elif isinstance(reply, TurnRequest):
                reply = reply.model_dump(exclude_unset=True)
            else:
                reply = deepcopy(reply)
            selected = reply.get('route', decision.get('agent'))
            if selected != decision.get('agent'):
                args = {}
            updates = reply.get('arguments', {})
            if selected == 'goals' and decision.get('action') == 'clarify' and 'absolute monthly contribution' in checked.get('question', '').lower():
                import re
                amount = re.fullmatch(r'\$?([\d,]+(?:\.\d{1,2})?)', reply.get('message', '').strip())
                if amount and 'goal_updates' not in updates:
                    updates = updates | {'goal_updates': {'monthly_contribution': float(amount[1].replace(',', ''))}}
            if updates.get('plan_id') and 'plan' not in updates:
                args.pop('plan', None)
            if selected == 'market':
                from workflow.router import extract_market_fields
                args.update(extract_market_fields(reply.get('message', '')))
            for key, value in updates.items():
                if key in {'goal_updates', 'plan_updates'} and isinstance(value, dict):
                    args[key] = args.get(key, {}) | value
                else:
                    args[key] = value
            request = TurnRequest(message=reply.get('message', 'Updated inputs'), route=selected, arguments=args,
                                  use_llm=reply.get('use_llm', original.use_llm), use_rag=reply.get('use_rag', original.use_rag))
            self.graph.invoke(Command(resume=request.model_dump(mode='json')), config)
            return self._response(thread_id)

    def cancel(self, thread_id):
        """Discard a pending request; retain previously validated context/reports."""
        with self._lock:
            config = self._config(thread_id)
            if self.graph.get_state(config).next:
                self.graph.update_state(config, {'pending_request': None, 'result': None,
                                                 'final_response': '', 'validated': {}}, as_node='finish')

    def _response(self, thread_id):
        snapshot = self.graph.get_state(self._config(thread_id))
        state = snapshot.values
        request = state['request']
        result = AgentResult.model_validate(state['result']) if state.get('result') else None
        metrics = dict(state.get('metrics', {})) | (result.metrics if result else {})
        decision = state.get('decision', {})
        metrics['routing_method'] = decision.get('method', 'rules')
        if snapshot.next:
            checked = state['validated']
            clarification = {k: checked[k] for k in ('question', 'missing', 'options') if k in checked}
            return TurnResponse(thread_id=thread_id, request_id=request['request_id'], status='needs_input',
                                message=checked['question'], agent=decision.get('agent'), clarification=clarification, metrics=metrics)
        status = 'error' if result and result.status == 'error' else (
            'unavailable' if result is None or result.status == 'unavailable' else 'complete')
        return TurnResponse(thread_id=thread_id, request_id=request['request_id'], status=status,
                            message=state['final_response'], agent=decision.get('agent'), result=result, metrics=metrics)
