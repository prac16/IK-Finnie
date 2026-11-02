"""Behavioral tests: routing, state isolation, interrupts, and real local analysis."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.goals.agent import GoalsPlanningAgent
from tools.alpha_vantage import AlphaVantageError
from tools.openai_router import OpenAIRouter, RoutingDraft
from tools.market_data import DailyBar, DailyHistory
from workflow.adapters import AgentAdapters
from workflow.artifacts import ArtifactStore
from workflow.router import route_request
from workflow.schemas import AgentResult, MarketArguments, TurnRequest
from workflow.service import WorkflowService
from workflow.validation import validate_request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'src' / 'data'
CSV = b'ticker,asset_type,market_value_usd,sector,sector_source\nAAPL,Stock,6000,Information Technology,User supplied\nBND,Bond ETF,4000,Not applicable,Non-equity instrument\n'


def plan():
    return json.loads((DATA / 'sample_goals_plan.json').read_text())


def request(message, **kwargs):
    return TurnRequest(message=message, use_llm=False, use_rag=False, **kwargs)


def draft(**updates):
    fields = dict(agent='chat', action='analyze', symbol=None, sessions=None, benchmark=None,
                  adjusted=None, goal_id=None, monthly_contribution=None, monthly_budget=None, question=None)
    return RoutingDraft(**(fields | updates))


def market_history(symbol='AAPL'):
    days = ('2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30')
    return DailyHistory(symbol, tuple(DailyBar(d, p, 1000) for d, p in zip(days, [100, 110, 99, 88, 105])),
                        'raw_close', '2026-11-01T00:00:00+00:00')


class RoutingTests(unittest.TestCase):
    def route(self, text, context=None, **kwargs):
        return route_request(request(text, **kwargs), context or {}, [], [])

    def test_labeled_rules(self):
        examples = [
            ('What is volatility?', 'chat'), ('What are drawdowns?', 'chat'),
            ('Analyze AAPL for 20 sessions against SPY', 'market'),
            ('Is my AAPL position too concentrated?', 'portfolio'),
            ('Analyze my holdings', 'portfolio'), ('Can I reach my house deposit goal?', 'goals'),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                result = self.route(text)
                self.assertEqual(result.agent, expected)
                self.assertEqual(result.router_calls, 0)

    def test_market_fields_basis_and_benchmark(self):
        result = self.route('Analyze AAPL for 30 sessions against SPY adjusted')
        self.assertEqual(result.arguments, dict(symbol='AAPL', sessions=30, benchmark='SPY', adjusted=True))
        self.assertFalse(self.route('Analyze AAPL unadjusted').arguments['adjusted'])

    def test_explicit_route_and_fields_win(self):
        result = self.route('Analyze AAPL for 20 sessions', route='market', arguments={'sessions': 10})
        self.assertEqual(result.arguments['sessions'], 10)
        self.assertEqual(result.method, 'explicit')

    def test_followup_requires_active_market(self):
        self.assertEqual(self.route('What about MSFT?').action, 'clarify')
        self.assertEqual(self.route('What about MSFT?', {'last_agent': 'market'}).agent, 'market')

    def test_saved_explanation_vs_concept(self):
        context = {'last_agent': 'market', 'reports': {'market': 'r'}}
        self.assertEqual(self.route('Explain that drawdown', context).action, 'explain')
        self.assertEqual(self.route('What is drawdown?', context).agent, 'chat')
        self.assertNotEqual(self.route('Explain that', {'last_agent': 'chat', 'reports': {'chat': 'r'}}).action, 'explain')

    def test_unsupported_and_composite(self):
        for message in ["Summarize today's news", 'Execute a buy order']:
            self.assertEqual(self.route(message).action, 'unsupported')
        self.assertEqual(self.route('Analyze my portfolio then evaluate my goal').action, 'clarify')

    def test_goal_edits_and_increment_clarification(self):
        result = self.route('What if I contribute 1100 monthly and set monthly budget to 1200?')
        self.assertEqual(result.agent, 'goals')
        self.assertEqual(result.arguments['goal_updates']['monthly_contribution'], 1100)
        self.assertEqual(result.arguments['plan_updates']['monthly_budget'], 1200)
        self.assertEqual(self.route('What if I contribute 200 more?').action, 'clarify')
        self.assertEqual(self.route('contribute 200 more', route='goals').action, 'clarify')

    def test_structured_arguments_select_agent(self):
        for args, expected in [({'plan_id': 'p'}, 'goals'), ({'portfolio_id': 'p'}, 'portfolio'), ({'symbol': 'AAPL'}, 'market')]:
            self.assertEqual(self.route('Please check this', arguments=args).agent, expected)

    def test_model_router_extracts_once(self):
        generator = Mock()
        generator.route.return_value = draft(agent='market', symbol='MSFT', sessions=15)
        result = route_request(TurnRequest(message='How has Microsoft performed?'), {}, [], [], generator=generator)
        self.assertEqual(result.arguments, {'symbol': 'MSFT', 'sessions': 15})
        self.assertEqual(result.router_calls, 1)
        generator.route.assert_called_once()

    def test_model_goal_updates_and_explicit_override(self):
        generator = Mock()
        generator.route.return_value = draft(agent='goals', goal_id='house', monthly_contribution=1100, monthly_budget=1200)
        result = route_request(TurnRequest(message='Change the funding'), {'plan_id': 'p'}, [], [], generator=generator)
        self.assertEqual(result.arguments['goal_id'], 'house')
        self.assertEqual(result.arguments['plan_updates']['monthly_budget'], 1200)

    def test_invalid_model_does_not_retry(self):
        for output in [{'agent': 'shell'}, draft(agent=None)]:
            generator = Mock()
            generator.route.return_value = output
            result = route_request(TurnRequest(message='Please help'), {}, [], [], generator=generator)
            self.assertEqual(result.action, 'clarify')
            self.assertEqual(result.router_calls, 1)
            generator.route.assert_called_once()

    def test_router_exception_is_sanitized(self):
        generator = Mock()
        generator.route.side_effect = RuntimeError('SECRET')
        result = route_request(TurnRequest(message='Please help'), {}, [], [], generator=generator)
        self.assertNotIn('SECRET', result.question)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name))

    def test_upload_and_cross_thread_access(self):
        meta = self.store.add_holdings('one', CSV, 'holdings.csv')[0]
        self.assertEqual(len(self.store.get('one', meta['id'], kind='portfolio')['holdings']), 2)
        self.assertEqual(len(meta['sha256']), 64)
        with self.assertRaises(ValueError): self.store.get('two', meta['id'])
        with self.assertRaises(ValueError): self.store.get('one', meta['id'], kind='goal_plan')
        with self.assertRaises(ValueError): self.store.metadata('one', '../bad')

    def test_immutable_versions_and_metadata_copies(self):
        original = self.store.add_plan('one', plan())
        updated = plan() | {'monthly_budget': 1000}
        new = self.store.add_plan('one', updated, parent_id=original['id'])
        self.assertEqual(new['version'], 2)
        self.assertEqual(self.store.get('one', original['id'])['monthly_budget'], 900)
        self.store.list('one')[0]['name'] = 'changed'
        self.assertEqual(self.store.metadata('one', original['id'])['name'], 'Goals plan')

    def test_size_count_and_integrity_limits(self):
        tiny = ArtifactStore(Path(self.tmp.name) / 'tiny', max_bytes=10, max_per_thread=1)
        with self.assertRaises(ValueError): tiny.put('t', 'x', {'long': 'abcdefghij'})
        tiny.put('t', 'x', {})
        with self.assertRaises(ValueError): tiny.put('t', 'x', {})
        meta = self.store.put('t', 'x', {})
        path = next(Path(self.tmp.name).glob('*/' + meta['id'] + '.json'))
        path.write_text('{"tampered":true}')
        with self.assertRaises(ValueError): self.store.get('t', meta['id'])
        with self.assertRaises(ValueError): self.store.native('t', meta['id'])

    def test_invalid_upload_is_not_registered(self):
        with self.assertRaises(ValueError): self.store.add_holdings('t', b'bad', 'holdings.csv')
        self.assertEqual(self.store.list('t'), [])
        with self.assertRaises(ValueError): self.store.add_plan('t', {})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name))
        self.provider = Mock()
        self.provider.get_history.side_effect = AlphaVantageError('No history')
        self.provider.get_quote.side_effect = AlphaVantageError('No quote')
        self.adapters = AgentAdapters(self.store, provider=self.provider)
        self.events = []
        self.service = WorkflowService(store=self.store, adapters=self.adapters, on_event=self.events.append)

    def send(self, text, **kwargs):
        return self.service.submit('one', request(text, **kwargs))

    def test_missing_ticker_interrupt_and_no_provider_work(self):
        result = self.send('Analyze market')
        self.assertEqual(result.status, 'needs_input')
        self.provider.get_history.assert_not_called()
        self.assertIn('symbol', result.message)
        self.assertTrue(self.service.graph.get_state(self.service._config('one')).next)
        with self.assertRaises(ValueError): self.send('new task')

    def test_invalid_resume_then_success_does_not_duplicate_calls(self):
        self.send('Analyze market')
        invalid = self.service.resume('one', {'message': 'Updated', 'arguments': {'symbol': 'AAPL', 'sessions': 1}})
        self.assertEqual(invalid.status, 'needs_input')
        self.provider.get_history.assert_not_called()
        result = self.service.resume('one', '20 sessions')
        self.assertEqual(result.agent, 'market')
        self.assertEqual(result.status, 'unavailable')
        self.provider.get_history.assert_called_once_with('AAPL', adjusted=False)
        self.assertEqual(result.metrics['provider_operations'], 2)
        self.assertEqual(result.metrics['router_calls'], 0)
        self.assertIsNone(self.service.get_state('one')['pending_request'])

    def test_cancel_then_switch_intent(self):
        self.send('Analyze market')
        self.service.cancel('one')
        result = self.send('What is volatility?')
        self.assertEqual(result.agent, 'chat')
        self.assertEqual(result.status, 'unavailable')
        self.service.cancel('one')
        with self.assertRaises(ValueError): self.service.resume('one', 'AAPL')

    def test_change_route_while_resuming(self):
        self.send('Analyze market')
        result = self.service.resume('one', request('What is volatility?', route='chat'))
        self.assertEqual(result.agent, 'chat')
        self.provider.get_history.assert_not_called()

    def test_real_portfolio_upload_metrics_and_provenance(self):
        portfolio = self.service.add_holdings('one', CSV, 'holdings.csv')[0]
        result = self.send('Analyze holdings')
        self.assertEqual(result.status, 'complete')
        self.assertEqual(result.result.input_id, portfolio['id'])
        report = self.service.get_report('one', result.result.report_id)['report']
        self.assertEqual(report['analysis']['calculated_value_usd'], 10000)
        self.assertEqual(report['analysis']['top_holding_pct'], 60)
        self.assertIn('Information Technology', report['analysis']['sector_allocation_pct'])
        self.assertEqual(result.metrics['narrative_calls'], 0)
        self.assertEqual(result.metrics['retrieval_calls'], 0)
        self.provider.get_history.assert_not_called()
        self.assertEqual(self.service.get_state('one')['active_context']['portfolio_id'], portfolio['id'])

    def test_upload_during_pending_clarification(self):
        self.assertEqual(self.send('Analyze holdings').status, 'needs_input')
        portfolio = self.service.add_holdings('one', CSV, 'holdings.csv')[0]
        result = self.service.resume('one', {'message': 'Use the upload', 'arguments': {'portfolio_id': portfolio['id']}})
        self.assertEqual(result.status, 'complete')

    def test_multiple_portfolios_require_selection(self):
        first = self.service.add_holdings('one', CSV, 'a.csv')[0]
        self.service.add_holdings('one', CSV, 'b.csv')
        result = self.send('Analyze portfolio')
        self.assertEqual(len(result.clarification['options']), 2)
        result = self.service.resume('one', {'arguments': {'portfolio_id': first['id']}})
        self.assertEqual(result.status, 'complete')

    def test_goal_real_calculations_and_budget_edits(self):
        self.service.add_plan('one', plan())
        result = self.send('Can I reach my goal?')
        self.assertEqual(result.status, 'complete')
        report = self.service.get_report('one', result.result.report_id)['report']
        self.assertEqual(report['results'][0]['projection']['ending_balance'], 47400)
        baseline_id = self.service.get_state('one')['active_context']['plan_id']
        invalid = self.send('What if I contribute 1100 monthly?')
        self.assertEqual(invalid.status, 'needs_input')
        self.assertIn('monthly_budget', invalid.message)
        self.assertEqual(self.service.get_state('one')['active_context']['plan_id'], baseline_id)
        revised = self.service.resume('one', {'arguments': {'plan_updates': {'monthly_budget': 1100}}})
        self.assertEqual(revised.status, 'complete')
        new_id = self.service.get_state('one')['active_context']['plan_id']
        self.assertNotEqual(new_id, baseline_id)
        self.assertEqual(self.store.get('one', new_id)['monthly_budget'], 1100)
        self.assertEqual(self.store.get('one', baseline_id)['monthly_budget'], 900)

    def test_goal_missing_fields_and_draft_completion(self):
        result = self.send('Plan my goal')
        self.assertEqual(result.status, 'needs_input')
        self.assertIn('annual_return_pct', json.dumps(plan()))
        result = self.service.resume('one', {'arguments': {'plan': plan()}})
        self.assertEqual(result.status, 'complete')

    def test_multiple_goal_plans_need_selection(self):
        self.service.add_plan('one', plan())
        second = self.service.add_plan('one', plan())
        result = self.send('Plan my goal')
        self.assertEqual(result.status, 'needs_input')
        self.assertEqual(len(result.clarification['options']), 2)
        self.assertEqual(self.service.resume('one', {'arguments': {'plan_id': second['id']}}).status, 'complete')

    def test_goal_edit_requires_selection_for_multiple_goals(self):
        inputs = json.loads((DATA / 'sample_multiple_goals.json').read_text())
        self.service.add_plan('one', inputs)
        self.send('Analyze goals')
        result = self.send('contribute 10 monthly')
        self.assertEqual(result.status, 'needs_input')
        self.assertIn('goal_id', result.message)

    def test_saved_report_explanation_reuses_evidence_without_retrieval(self):
        self.service.add_plan('one', plan())
        first = self.send('Analyze goal')
        result = self.send('Explain that shortfall')
        self.assertEqual(result.status, 'complete')
        self.assertTrue(result.metrics['reused_report'])
        self.assertEqual(result.metrics['retrieval_calls'], 0)
        self.assertEqual(result.result.input_id, first.result.input_id)
        self.assertIn('not refreshed', result.message)
        self.assertEqual(self.store.metadata('one', result.result.report_id)['parent_id'], first.result.report_id)

    def test_two_threads_are_isolated(self):
        p = self.service.add_holdings('one', CSV, 'holdings.csv')[0]
        self.send('Analyze portfolio')
        result = self.service.submit('two', request('Analyze portfolio', arguments={'portfolio_id': p['id']}))
        self.assertEqual(result.status, 'needs_input')
        self.assertNotIn('portfolio_id', self.service.get_state('two')['active_context'])
        self.assertEqual(len(self.service.get_state('two')['messages']), 1)
        with self.assertRaises(ValueError): self.service.get_report('two', self.service.get_state('one')['result']['report_id'])

    def test_state_resets_results_between_turns(self):
        self.service.add_plan('one', plan())
        self.send('Analyze goals')
        result = self.send('Analyze portfolio')
        state = self.service.get_state('one')
        self.assertEqual(result.status, 'needs_input')
        self.assertIsNone(state['result'])
        self.assertEqual(state['final_response'], '')
        self.assertEqual(state['metrics']['execution_ms'], 0)

    def test_new_upload_replaces_selected_input_on_next_turn(self):
        first = self.service.add_holdings('one', CSV, 'one.csv')[0]
        self.send('Analyze holdings')
        second = self.service.add_holdings('one', CSV.replace(b'6000', b'12000'), 'two.csv')[0]
        result = self.send('Analyze these holdings')
        self.assertNotEqual(first['id'], result.result.input_id)
        self.assertEqual(second['id'], result.result.input_id)
        report = self.service.get_report('one', result.result.report_id)['report']
        self.assertEqual(report['analysis']['calculated_value_usd'], 16000)

    def test_new_goal_plan_replaces_selected_goal_context(self):
        self.service.add_plan('one', plan())
        self.send('Analyze goals')
        revised = plan()
        revised['goals'][0]['id'] = 'different'
        new = self.service.add_plan('one', revised)
        result = self.send('Analyze goals')
        self.assertEqual(self.service.get_state('one')['active_context']['goal_ids'], ['different'])
        self.assertEqual(self.store.metadata('one', result.result.input_id)['parent_id'], new['id'])

    def test_unsupported_is_terminal_without_specialist(self):
        result = self.send('Summarize the news')
        self.assertEqual(result.status, 'unavailable')
        self.assertIsNone(result.result)
        self.provider.get_history.assert_not_called()

    def test_invalid_market_inputs_and_unknown_arguments(self):
        for args in [{'symbol': 'AAPL', 'sessions': True}, {'symbol': 'AAPL', 'benchmark': 'AAPL'},
                     {'symbol': '../../bad'}, {'symbol': 'AAPL', 'shell': 'anything'}]:
            result = self.send('Analyze market', route='market', arguments=args)
            self.assertEqual(result.status, 'needs_input')
            self.service.cancel('one')
        self.provider.get_history.assert_not_called()

    def test_thread_id_and_blank_message_validation(self):
        for value in ['', ' ', 'x' * 129]:
            with self.assertRaises(ValueError): self.service.get_state(value)
        with self.assertRaises(ValueError): TurnRequest(message=' ')

    def test_adapter_exception_sanitized_and_previous_context_retained(self):
        self.service.add_plan('one', plan())
        self.send('Analyze goals')
        before = self.service.get_state('one')['active_context']
        with patch.object(self.adapters, 'execute', side_effect=RuntimeError('SECRET')):
            result = self.send('Analyze AAPL')
        self.assertEqual(result.status, 'error')
        self.assertNotIn('SECRET', result.message)
        self.assertEqual(before, self.service.get_state('one')['active_context'])

    def test_events_contain_no_financial_payload(self):
        self.send('Analyze market')
        self.assertEqual(self.events[0]['stage'], 'routing')
        self.assertEqual(set(self.events[0]), {'stage', 'request_id'})


class RouterTransportTests(unittest.TestCase):
    def test_strict_schema_and_one_call(self):
        with patch('tools.openai_router.OpenAI') as client:
            model = OpenAIRouter()
            choice = SimpleNamespace(finish_reason='stop', message=SimpleNamespace(refusal=None, content=draft().model_dump_json()))
            client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
            self.assertEqual(model.route({'message': 'help'}).agent, 'chat')
            kwargs = client.return_value.chat.completions.create.call_args.kwargs
            self.assertTrue(kwargs['response_format']['json_schema']['strict'])
            self.assertFalse(kwargs['store'])
            self.assertEqual(client.call_args.kwargs['max_retries'], 0)

    def test_empty_and_refused_output(self):
        with patch('tools.openai_router.OpenAI') as client:
            model = OpenAIRouter()
            for choices in [[], [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(refusal='no', content=None))]]:
                client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=choices)
                with self.assertRaises(ValueError): model.route({})


class AdapterIntegrationTests(unittest.TestCase):
    # These tests execute the existing specialist implementations with injected
    # providers/generators, so assertions cover behavior across real boundaries.
    def setUp(self):
        WorkflowTests.setUp(self)
        self.collection = Mock()
        self.collection.query.return_value = {
            'documents': [['Diversification can reduce concentration risk.']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 2, 'category': 'investing'}]],
            'distances': [[0.2]],
        }

    send = WorkflowTests.send

    def test_market_followup_reuses_settings_and_tracks_provider_calls(self):
        self.provider.get_history.side_effect = lambda symbol, **kwargs: market_history(symbol)
        first = self.send('Analyze AAPL for 4 sessions against SPY')
        self.assertEqual(first.status, 'complete')
        self.assertEqual(first.metrics['provider_operations'], 2)
        second = self.send('What about MSFT?')
        self.assertEqual(second.status, 'complete')
        report = self.service.get_report('one', second.result.report_id)['report']
        self.assertEqual(report['symbol'], 'MSFT')
        self.assertEqual(report['requested_sessions'], 4)
        self.assertEqual(report['benchmark_symbol'], 'SPY')
        self.assertAlmostEqual(report['metrics']['period_return_pct'], 5)
        self.assertEqual(self.service.submit('two', request('What about MSFT?')).status, 'needs_input')

    def test_market_partial_and_unavailable_do_not_leave_stale_report(self):
        self.provider.get_history.side_effect = [market_history(), AlphaVantageError('fail')]
        first = self.send('Analyze AAPL for 4 sessions against SPY')
        self.assertEqual(first.result.status, 'partial')
        self.provider.get_history.side_effect = AlphaVantageError('fail')
        unavailable = self.send('What about MSFT?')
        self.assertEqual(unavailable.status, 'unavailable')
        explanation = self.send('Explain that drawdown')
        self.assertEqual(explanation.status, 'unavailable')
        self.assertIn('MSFT', explanation.message)
        self.assertTrue(explanation.metrics['reused_report'])

    def test_chat_local_excerpts_and_single_retrieval(self):
        self.adapters.collection_factory = lambda: self.collection
        result = self.service.submit('one', TurnRequest(message='What is diversification?', use_llm=False))
        self.assertEqual(result.status, 'complete')
        self.assertIn('Local excerpts', result.message)
        self.assertIn('Risk.pdf', result.message)
        self.assertEqual(result.metrics['retrieval_calls'], 1)
        self.assertEqual(result.metrics['narrative_calls'], 0)

    def test_chat_one_llm_and_no_composer_call(self):
        self.adapters.collection_factory = lambda: self.collection
        generator = Mock(model='test-model')
        generator.generate.return_value = 'Diversification can reduce risk [1].'
        self.adapters.generators['chat'] = lambda: generator
        result = self.service.submit('one', TurnRequest(message='What is diversification?'))
        self.assertEqual(result.status, 'complete')
        self.assertEqual(result.metrics['narrative_calls'], 1)
        self.assertEqual(result.metrics['retrieval_calls'], 1)
        generator.generate.assert_called_once()

    def test_rag_failure_preserves_goal_metrics_and_does_not_leak(self):
        self.adapters.collection_factory = Mock(side_effect=RuntimeError('SECRET'))
        result = self.service.submit('one', TurnRequest(message='Analyze goals', arguments={'plan': plan()}, use_llm=False))
        self.assertEqual(result.status, 'complete')
        self.assertIn('47,400', result.message)
        self.assertIn('RAG index unavailable', result.message)
        self.assertNotIn('SECRET', result.message)

    def test_one_goal_retrieval_and_one_generation_then_focused_followup(self):
        self.adapters.collection_factory = lambda: self.collection
        generator = Mock(model='test-model')
        # Deliberately invalid output exercises the existing labeled fallback.
        generator.generate.side_effect = ValueError('SECRET')
        self.adapters.generators['goals'] = lambda: generator
        result = self.service.submit('one', TurnRequest(message='Analyze goals', arguments={'plan': plan()}))
        self.assertEqual(result.status, 'complete')
        self.assertEqual(result.metrics['retrieval_calls'], 1)
        self.assertEqual(result.metrics['narrative_calls'], 1)
        self.assertIn('Local fallback', result.message)
        followup = self.service.submit('one', TurnRequest(message='Explain that shortfall'))
        self.assertEqual(followup.metrics['retrieval_calls'], 0)
        self.assertEqual(followup.metrics['narrative_calls'], 1)
        self.assertEqual(generator.generate.call_args.args[0]['user_question'], 'Explain that shortfall')
        self.collection.query.assert_called_once()
        self.assertNotIn('SECRET', followup.message)

    def test_portfolio_followup_focuses_on_saved_report(self):
        self.service.add_holdings('one', CSV, 'holdings.csv')
        generator = Mock(model='test-model')
        generator.generate.side_effect = ValueError('bad output')
        self.adapters.generators['portfolio'] = lambda: generator
        self.send('Analyze holdings')
        result = self.service.submit('one', TurnRequest(message='Explain that concentration', use_rag=False))
        self.assertEqual(result.status, 'complete')
        self.assertEqual(result.metrics['narrative_calls'], 1)
        self.assertEqual(generator.generate.call_args.args[0]['user_question'], 'Explain that concentration')

    def test_market_followup_focuses_without_provider_or_rag_calls(self):
        self.provider.get_history.side_effect = lambda symbol, **kwargs: market_history(symbol)
        generator = Mock(model='test-model')
        generator.generate.side_effect = ValueError('bad output')
        self.adapters.generators['market'] = lambda: generator
        self.send('Analyze AAPL for 4 sessions')
        before = self.provider.get_history.call_count
        result = self.service.submit('one', TurnRequest(message='Explain that drawdown', use_rag=False))
        self.assertEqual(result.metrics['provider_operations'], 0)
        self.assertEqual(result.metrics['retrieval_calls'], 0)
        self.assertEqual(result.metrics['narrative_calls'], 1)
        self.assertEqual(self.provider.get_history.call_count, before)
        self.assertEqual(generator.generate.call_args.args[0]['user_question'], 'Explain that drawdown')

    def test_api_failure_counts_operations_and_sanitizes(self):
        self.provider.get_history.side_effect = RuntimeError('SECRET')
        result = self.send('Analyze AAPL')
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.metrics['provider_operations'], 1)
        self.assertNotIn('SECRET', result.message)

    def test_invalid_chat_category_clarifies(self):
        result = self.send('What is volatility?', arguments={'category': 'unknown'})
        self.assertEqual(result.status, 'needs_input')

    def test_increment_reply_preserves_budget_constraint(self):
        self.service.add_plan('one', plan())
        self.send('Analyze goals')
        self.assertEqual(self.send('What if I contribute 200 more?').status, 'needs_input')
        result = self.service.resume('one', '1100')
        self.assertEqual(result.status, 'needs_input')
        self.assertIn('monthly_budget', result.message)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name) / 'runtime')
        self.service = WorkflowService(store=self.store)

    def run_cli(self, args, replies=()):
        from workflow.cli import main
        with patch('sys.argv', ['finnie-chat', *args]), patch('workflow.cli.WorkflowService', return_value=self.service), \
             patch('builtins.input', side_effect=list(replies)), patch('builtins.print') as output:
            main()
        return output

    def test_single_goal_command_json(self):
        output = self.run_cli(['Analyze goals', '--plan', str(DATA / 'sample_goals_plan.json'), '--no-rag', '--no-llm', '--json'])
        response = json.loads(output.call_args.args[0])
        self.assertEqual(response['status'], 'complete')
        self.assertEqual(response['metrics']['narrative_calls'], 0)

    def test_interactive_upload_then_analyze_and_cancel(self):
        path = Path(self.tmp.name) / 'holdings.csv'
        path.write_bytes(CSV)
        self.run_cli(['--no-rag', '--no-llm', '--thread-id', 'one'], [
            '/upload ' + str(path), '/artifacts', 'Analyze holdings', 'Analyze market', '/cancel', '', '/quit'])
        state = self.service.get_state('one')
        self.assertIn('portfolio_id', state['active_context'])
        self.assertIsNone(state['pending_request'])

    def test_interactive_plan_and_json_update(self):
        self.run_cli(['--no-rag', '--no-llm', '--thread-id', 'one'], [
            '/plan ' + str(DATA / 'sample_goals_plan.json'), 'Analyze goals',
            '{"goal_updates":{"monthly_contribution":100},"goal_id":"house"}', '/quit'])
        state = self.service.get_state('one')
        self.assertEqual(self.store.get('one', state['active_context']['plan_id'])['goals'][0]['monthly_contribution'], 100)

    def test_needs_input_exit_status_and_input_error(self):
        with self.assertRaises(SystemExit) as exc:
            self.run_cli(['Analyze market', '--no-rag', '--no-llm'])
        self.assertEqual(exc.exception.code, 2)
        self.run_cli(['--no-rag', '--no-llm'], ['{bad json', '/quit'])

    def test_arguments_file_and_portfolio_flag(self):
        path = Path(self.tmp.name) / 'holdings.csv'
        path.write_bytes(CSV)
        arguments = Path(self.tmp.name) / 'args.json'
        arguments.write_text('{}')
        self.run_cli(['Analyze holdings', '--holdings', str(path), '--arguments', str(arguments),
                      '--route', 'portfolio', '--no-rag', '--no-llm'])

    def test_invalid_json_file_type_and_large_json(self):
        from workflow.cli import _read_json
        path = Path(self.tmp.name) / 'args.json'
        path.write_text('[]')
        with patch('sys.stderr'), self.assertRaises(SystemExit): self.run_cli(['hello', '--arguments', str(path)])
        path.write_bytes(b'x' * (1024 * 1024 + 1))
        with self.assertRaises(ValueError): _read_json(path)


if __name__ == '__main__':
    unittest.main()
