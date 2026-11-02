"""Session-scoped controller; Streamlit reruns never execute requests implicitly."""
from __future__ import annotations

import csv
import io
import hashlib
from uuid import uuid4

from agents.goals.models import GoalPlan
from workflow import TurnRequest, WorkflowService
from workflow.artifacts import ArtifactStore
from ui.memory import ProfileStore, UserProfile
from tools.llm_privacy import privacy_scope


def portfolio_csv(portfolio: dict) -> str:
    buffer = io.StringIO()
    columns = ['ticker', 'asset_type', 'asset_class', 'market_value_usd', 'sector', 'sector_source',
               'portfolio', 'as_of_date', 'age', 'risk_tolerance', 'goal', 'analysis_focus']
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for holding in portfolio['holdings']:
        writer.writerow({**{k: holding.get(k) for k in columns if k in holding},
                         'asset_class': holding.get('asset_class_override'), 'portfolio': portfolio['name'],
                         **{k: portfolio.get(k) for k in ['as_of_date', 'age', 'risk_tolerance', 'goal', 'analysis_focus']}})
    return buffer.getvalue()


class SessionMemory:
    """One instance per Streamlit session, never a globally cached resource."""
    def __init__(self, *, profiles=None, service_factory=None):
        self.profiles = profiles if profiles is not None else ProfileStore()
        self.service_factory = service_factory or (lambda: WorkflowService(store=ArtifactStore(persist=False)))
        self.profile = UserProfile(name='Guest')
        self.profile_key = 'guest'
        self.reset_conversation()

    def reset_conversation(self):
        self.service = self.service_factory()
        self.thread_id = uuid4().hex
        self.chat_service = self.service_factory()
        self.chat_thread_id = uuid4().hex
        self.messages = []
        self.analysis_messages = []
        self.latest = {}
        self.goal_results = {}
        self.uploads = {}
        self.selected_portfolio = None
        self.selected_plan = None

    def switch_profile(self, profile_id):
        profile = self.profiles.load(profile_id) if profile_id != 'guest' else UserProfile(name='Guest')
        self.profile, self.profile_key = profile, profile_id
        self.reset_conversation()

    def create_profile(self, name):
        profile = self.profiles.save(UserProfile(name=name))
        self.switch_profile(profile.id)
        return profile.id

    def save_preferences(self, **updates):
        if self.profile_key == 'guest':
            raise ValueError('Create a local profile to save preferences')
        allowed = {'name', 'currency', 'risk_preference', 'use_llm', 'use_rag'}
        if not set(updates) <= allowed:
            raise ValueError('Unsupported preference')
        profile = UserProfile.model_validate(self.profile.model_dump() | updates)
        self.profile = self.profiles.save(profile)
        return self.profile

    def add_holdings(self, content: bytes, filename: str):
        digest = hashlib.sha256(content).hexdigest()
        if digest not in self.uploads:
            self.uploads[digest] = self.service.add_holdings(self.thread_id, content, filename)
        options = self.uploads[digest]
        self.selected_portfolio = options[0]['id'] if len(options) == 1 else None
        return options

    def add_plan(self, inputs: dict):
        plan = GoalPlan.model_validate(inputs).model_dump(mode='json')
        meta = self.service.add_plan(self.thread_id, plan)
        self.selected_plan = meta['id']
        return meta

    def ask(self, message, *, route=None, arguments=None, use_llm=None, use_rag=None, resume=None):
        request = TurnRequest(message=message, route=route, arguments=arguments or {},
                              use_llm=self.profile.use_llm if use_llm is None else use_llm,
                              use_rag=self.profile.use_rag if use_rag is None else use_rag)
        pending = self.service.get_state(self.thread_id).get('pending_request')
        if resume is False and pending:
            self.service.cancel(self.thread_id)
            pending = None
        with privacy_scope([self.profile.name] if self.profile_key != 'guest' else []):
            return self._dispatch(request, pending, route)

    def _dispatch(self, request, pending, route):
        message = request.message
        if pending:
            # Do not send an explicit None route: a clarification inherits its agent.
            reply = request.model_dump(mode='json', exclude={'route'})
            if route is not None:
                reply['route'] = route
            response = self.service.resume(self.thread_id, reply)
        else:
            response = self.service.submit(self.thread_id, request)
        self.analysis_messages.extend([{'role': 'user', 'text': message},
                              {'role': 'assistant', 'response': response.model_dump(mode='json')}])
        if response.agent:
            self.latest[response.agent] = response.model_dump(mode='json')
        state = self.service.get_state(self.thread_id)
        context = state.get('active_context', {})
        self.selected_portfolio = context.get('portfolio_id', self.selected_portfolio)
        self.selected_plan = context.get('plan_id', self.selected_plan)
        return response

    def ask_chat(self, question):
        """Knowledge-base Q&A only: no analytical inputs, routing or report context."""
        request = TurnRequest(message=question, route='chat',
                              arguments={'question': question}, use_llm=True, use_rag=True)
        with privacy_scope([self.profile.name] if self.profile_key != 'guest' else []):
            response = self.chat_service.submit(self.chat_thread_id, request)
        self.messages.extend([{'role': 'user', 'text': question},
                              {'role': 'assistant', 'response': response.model_dump(mode='json')}])
        return response

    def save_holdings(self, portfolio_id):
        if self.profile_key == 'guest':
            raise ValueError('Create a local profile before saving holdings')
        portfolio = self.service.store.get(self.thread_id, portfolio_id, kind='portfolio')
        profile = self.profile.model_copy(update={'saved_holdings_csv': portfolio_csv(portfolio),
                                                  'saved_holdings_name': portfolio['name']})
        self.profile = self.profiles.save(profile)
        return self.profile

    def save_plan(self, plan_id):
        if self.profile_key == 'guest':
            raise ValueError('Create a local profile before saving a goals plan')
        plan = GoalPlan.model_validate(self.service.store.get(self.thread_id, plan_id, kind='goal_plan'))
        self.profile = self.profiles.save(self.profile.model_copy(update={'saved_plan': plan}))
        return self.profile

    def restore_holdings(self):
        if not self.profile.saved_holdings_csv:
            raise ValueError('No holdings are saved for this profile')
        return self.add_holdings(self.profile.saved_holdings_csv.encode(), 'saved_holdings.csv')

    def restore_plan(self):
        if self.profile.saved_plan is None:
            raise ValueError('No goals plan is saved for this profile')
        return self.add_plan(self.profile.saved_plan.model_dump(mode='json'))

    def forget_saved_inputs(self):
        if self.profile_key == 'guest':
            return
        self.profile = self.profiles.save(self.profile.model_copy(update={
            'saved_plan': None, 'saved_holdings_csv': None, 'saved_holdings_name': None}))
