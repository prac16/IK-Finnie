"""Adapters preserve specialist calculations, evidence and fallback behavior."""
from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter
from typing import Protocol

from workflow.schemas import AgentResult, TurnRequest
from tools.llm_runtime import generate_checked
from tools.diagnostics import log_event


class SpecialistAdapter(Protocol):
    def execute(self, agent: str, validated: dict, request: TurnRequest, thread_id: str) -> AgentResult: ...


class _MeasuredCollection:
    def __init__(self, collection, metrics):
        self.collection, self.metrics = collection, metrics

    def query(self, **kwargs):
        self.metrics['retrieval_calls'] += 1
        return self.collection.query(**kwargs)


class _MeasuredGenerator:
    def __init__(self, factory, metrics):
        self.factory, self.metrics, self.delegate = factory, metrics, None

    @property
    def model(self):
        return getattr(self.delegate, 'model', None)

    def generate(self, *args, **kwargs):
        self.delegate = self.delegate or self.factory()
        try:
            return self.delegate.generate(*args, **kwargs)
        finally:
            count = getattr(self.delegate, 'attempt_count', 1)
            self.metrics['narrative_calls'] += count if type(count) is int else 1

    def generate_validated(self, *args, validate=None, **kwargs):
        self.delegate = self.delegate or self.factory()
        try:
            return generate_checked(self.delegate, *args, validate=validate, **kwargs)
        finally:
            count = getattr(self.delegate, 'attempt_count', 1)
            self.metrics['narrative_calls'] += count if type(count) is int else 1


class _MeasuredProvider:
    def __init__(self, provider, metrics):
        self.provider, self.metrics = provider, metrics

    def get_history(self, *args, **kwargs):
        self.metrics['provider_operations'] += 1
        return self.provider.get_history(*args, **kwargs)

    def get_quote(self, *args, **kwargs):
        self.metrics['provider_operations'] += 1
        return self.provider.get_quote(*args, **kwargs)


class _ExcerptGenerator:
    def generate(self, *, system, prompt):
        return 'Local excerpts; no LLM interpretation was generated.\n\n' + prompt.split('Retrieved document excerpts:\n', 1)[-1]


class AgentAdapters:
    def __init__(self, store, *, collection_factory=None, provider=None, generators=None):
        self.store, self.provider = store, provider
        self.collection_factory = collection_factory
        self.generators = generators or {}
        self._collection = None

    def _generator(self, agent, metrics):
        if agent in self.generators:
            return _MeasuredGenerator(self.generators[agent], metrics)
        if agent == 'portfolio':
            from tools.openai_explanation import OpenAIExplanationGenerator
            factory = OpenAIExplanationGenerator
        elif agent == 'market':
            from tools.openai_market import OpenAIMarketExplanationGenerator
            factory = OpenAIMarketExplanationGenerator
        elif agent == 'goals':
            from tools.openai_goals import OpenAIGoalsExplanationGenerator
            factory = OpenAIGoalsExplanationGenerator
        else:
            from tools.openai_generation import OpenAIGenerator
            import os
            factory = lambda: OpenAIGenerator(model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
        return _MeasuredGenerator(factory, metrics)

    def execute(self, agent, validated, request, thread_id):
        started = perf_counter()
        metrics = {'narrative_calls': 0, 'retrieval_calls': 0, 'provider_operations': 0,
                   'reused_report': validated['action'] == 'explain'}
        try:
            return self._execute(agent, validated, request, thread_id, started, metrics)
        except Exception as exc:
            log_event('analysis_failed', agent=agent, error_type=type(exc).__name__)
            metrics['elapsed_ms'] = round((perf_counter() - started) * 1000, 2)
            return AgentResult(agent=agent, request_id=request.request_id, status='error', metrics=metrics,
                               markdown='The analysis could not be completed. Check your inputs and provider configuration, then try again.')

    def _execute(self, agent, validated, request, thread_id, started, metrics):
        warnings = []
        generator = self._generator(agent, metrics) if request.use_llm else None
        collection = None
        # Saved-report explanations never open an index or retrieve again.
        if request.use_rag and validated['action'] != 'explain':
            try:
                if self._collection is None:
                    from rag.store import local_collection
                    self._collection = (self.collection_factory or local_collection)()
                collection = _MeasuredCollection(self._collection, metrics)
            except Exception:
                warnings.append('Local RAG index unavailable; no knowledge excerpts attached.')
        args = validated['arguments']
        input_id = validated.get('input_id')
        if validated['action'] == 'explain':
            report = self.store.native(thread_id, validated['report_id'])
            kwargs = dict(use_llm=request.use_llm, generator=generator, user_question=request.message)
            if agent == 'portfolio':
                from agents.portfolio.explanation import explain_report
                explanation = explain_report(report, **kwargs)
            elif agent == 'market':
                from agents.market.explanation import explain_market_report
                explanation = explain_market_report(report, **kwargs)
            elif agent == 'goals':
                from agents.goals.explanation import explain_goals_report
                explanation = explain_goals_report(report, **kwargs)
            else:
                raise ValueError('Choose a saved analytical report to explain')
            report = replace(report, explanation=explanation)
            saved = self.store.get(thread_id, validated['report_id'], kind='report')
            input_id = saved.get('input_id')
            # Keep the underlying report and its linked evidence available.
            markdown = 'Based on the saved report; data and assumptions were not refreshed.\n\n' + report.to_markdown()
        elif agent == 'portfolio':
            from agents.portfolio.report import build_report
            report = build_report(self.store.native(thread_id, input_id), collection,
                                  use_llm=request.use_llm, explanation_generator=generator)
            markdown = report.to_markdown()
        elif agent == 'market':
            from agents.market.agent import MarketAnalysisAgent
            if self.provider is None:
                from tools.market_data import AlphaVantageMarketData
                self.provider = AlphaVantageMarketData()
            report = MarketAnalysisAgent(_MeasuredProvider(self.provider, metrics), collection=collection,
                                        use_llm=request.use_llm, generator=generator).analyze(**args)
            markdown = report.to_markdown()
        elif agent == 'goals':
            from agents.goals.agent import GoalsPlanningAgent
            report = GoalsPlanningAgent(collection=collection, use_llm=request.use_llm, generator=generator).plan(args['plan'])
            input_id = self.store.add_plan(thread_id, args['plan'], parent_id=validated.get('parent_id'))['id']
            markdown = report.to_markdown()
        else:
            if collection is None:
                return AgentResult(agent='chat', request_id=request.request_id, status='unavailable',
                                   markdown='Financial Q&A requires the local knowledge index. Enable RAG and check the index.',
                                   warnings=warnings, metrics=metrics)
            from agents.chat.agent import QnAAgent
            report = QnAAgent(collection, generator or _ExcerptGenerator(), category=args['category']).answer(args['question'])
            markdown = report.to_markdown()
        payload = report.to_dict() if hasattr(report, 'to_dict') else asdict(report)
        meta = self.store.put(thread_id, 'report', {'agent': agent, 'input_id': input_id, 'report': payload},
                              name=agent + ' report', native=report,
                              parent_id=validated.get('report_id'))
        if warnings:
            markdown += '\n\nWorkflow warnings:\n' + '\n'.join('- ' + w for w in warnings)
        warnings.extend(getattr(report, 'warnings', ()))
        explanation = getattr(report, 'explanation', None)
        metrics.update(elapsed_ms=round((perf_counter() - started) * 1000, 2),
                       explanation_method=getattr(explanation, 'method', None),
                       explanation_model=getattr(explanation, 'model', None))
        raw_status = getattr(report, 'status', 'complete')
        status = {'complete': 'success', 'partial': 'partial', 'unavailable': 'unavailable'}.get(raw_status, 'success')
        if explanation and getattr(explanation, 'fallback_reason', None) and status == 'success':
            status = 'partial'
        return AgentResult(agent=agent, request_id=request.request_id, status=status, markdown=markdown,
                           report_id=meta['id'], input_id=input_id, warnings=warnings, metrics=metrics)
