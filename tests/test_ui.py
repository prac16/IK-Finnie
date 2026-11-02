"""Session isolation, opt-in user persistence, and Streamlit form interactions."""
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest
from agents.portfolio.holdings import load_holdings
from tools.alpha_vantage import AlphaVantageError
from tools.market_data import DailyBar, DailyHistory
from ui.controller import SessionMemory, portfolio_csv
from ui.memory import ProfileStore, UserProfile
from workflow import WorkflowService
from workflow.adapters import AgentAdapters
from workflow.artifacts import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
CSV = b'ticker,asset_type,market_value_usd,sector,sector_source,portfolio,as_of_date,goal\nAAPL,Stock,6000,Information Technology,User supplied,My holdings,2026-11-01,House deposit\nBND,Bond ETF,4000,Not applicable,Non-equity instrument,My holdings,2026-11-01,House deposit\n'


def sample_plan():
    return json.loads((ROOT / 'src/data/sample_goals_plan.json').read_text())


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles = ProfileStore(Path(self.tmp.name) / 'users')
        self.memory = SessionMemory(profiles=self.profiles)
        self.memory.profile.use_llm = False
        self.memory.profile.use_rag = False

    def test_guest_session_inputs_and_reports_never_write_to_disk(self):
        self.memory.add_holdings(CSV, 'holdings.csv')
        response = self.memory.ask('Analyze holdings', route='portfolio')
        self.assertEqual(response.status, 'complete')
        self.assertEqual(list(Path(self.tmp.name).rglob('*')), [])
        self.assertFalse(self.memory.service.store.persist)
        with self.assertRaises(ValueError): self.memory.save_holdings(self.memory.selected_portfolio)

    def test_new_conversation_clears_context_but_preserves_profile(self):
        profile_id = self.memory.create_profile('Local user')
        self.memory.save_preferences(use_llm=False, use_rag=False)
        self.memory.add_plan(sample_plan())
        response = self.memory.ask('Analyze goals', route='goals', arguments={'plan_id': self.memory.selected_plan})
        self.memory.save_plan(response.result.input_id)
        previous_thread = self.memory.thread_id
        self.memory.reset_conversation()
        self.assertNotEqual(previous_thread, self.memory.thread_id)
        self.assertEqual(self.memory.messages, [])
        self.assertEqual(self.memory.service.store.list(self.memory.thread_id), [])
        self.assertEqual(self.memory.profile.id, profile_id)
        self.assertIsNotNone(self.memory.profile.saved_plan)
        self.assertEqual(self.memory.service.get_state(self.memory.thread_id), {})

    def test_saved_inputs_restore_in_fresh_process_controller(self):
        profile_id = self.memory.create_profile('Saver')
        self.memory.save_preferences(use_llm=False, use_rag=False, currency='USD', risk_preference='low')
        portfolio_id = self.memory.add_holdings(CSV, 'holdings.csv')[0]['id']
        self.memory.save_holdings(portfolio_id)
        self.memory.add_plan(sample_plan())
        self.memory.save_plan(self.memory.selected_plan)
        fresh = SessionMemory(profiles=ProfileStore(self.profiles.root))
        fresh.switch_profile(profile_id)
        self.assertEqual(fresh.messages, [])
        self.assertEqual(fresh.service.store.list(fresh.thread_id), [])
        fresh.restore_holdings()
        fresh.restore_plan()
        report = fresh.ask('Analyze holdings', route='portfolio')
        saved = fresh.service.get_report(fresh.thread_id, report.result.report_id)['report']
        self.assertEqual(saved['analysis']['calculated_value_usd'], 10000)
        self.assertEqual(saved['analysis']['portfolio']['as_of_date'], '2026-11-01')
        self.assertEqual(fresh.profile.risk_preference, 'low')

    def test_profile_switch_clears_uploads_messages_and_access(self):
        one = self.memory.create_profile('One')
        p = self.memory.add_holdings(CSV, 'holdings.csv')[0]
        self.memory.save_holdings(p['id'])
        old_thread = self.memory.thread_id
        two = self.memory.create_profile('Two')
        self.assertNotEqual(one, two)
        self.assertNotEqual(old_thread, self.memory.thread_id)
        self.assertIsNone(self.memory.profile.saved_holdings_csv)
        with self.assertRaises(ValueError): self.memory.service.store.get(self.memory.thread_id, p['id'])
        self.memory.switch_profile(one)
        self.assertIsNotNone(self.memory.profile.saved_holdings_csv)

    def test_forget_inputs_preserves_preferences_and_live_session(self):
        profile_id = self.memory.create_profile('Saver')
        self.memory.save_preferences(currency='USD')
        self.memory.add_plan(sample_plan())
        self.memory.save_plan(self.memory.selected_plan)
        self.memory.forget_saved_inputs()
        profile = self.profiles.load(profile_id)
        self.assertIsNone(profile.saved_plan)
        self.assertEqual(profile.currency, 'USD')
        self.assertIsNotNone(self.memory.selected_plan)

    def test_duplicate_uploads_do_not_consume_more_artifacts(self):
        first = self.memory.add_holdings(CSV, 'one.csv')
        second = self.memory.add_holdings(CSV, 'two.csv')
        self.assertEqual(first, second)
        self.assertEqual(len(self.memory.service.store.list(self.memory.thread_id)), 1)

    def test_unsaved_inputs_do_not_enter_user_memory(self):
        identifier = self.memory.create_profile('Unsaved')
        self.memory.add_holdings(CSV, 'holdings.csv')
        self.memory.add_plan(sample_plan())
        persisted = json.loads((self.profiles.root / (identifier + '.json')).read_text())
        self.assertIsNone(persisted['saved_plan'])
        self.assertIsNone(persisted['saved_holdings_csv'])
        self.assertNotIn('messages', persisted)
        self.assertNotIn('reports', persisted)

    def test_profiles_validate_ids_fields_and_corrupted_files(self):
        with self.assertRaises(ValueError): self.profiles.load('../secret')
        with self.assertRaises(ValueError): self.profiles.save(UserProfile(name=' '))
        self.profiles.save(UserProfile(name='Valid'))
        (self.profiles.root / 'invalid.json').write_text('{}')
        self.assertEqual(len(self.profiles.list()), 1)
        with self.assertRaises(ValueError): self.memory.save_preferences(currency='EUR')
        identifier = self.memory.create_profile('USD profile')
        with self.assertRaises(ValueError): self.memory.save_preferences(currency='EUR')
        self.assertEqual(self.profiles.load(identifier).currency, 'USD')
        foreign = sample_plan() | {'currency': 'EUR'}
        with self.assertRaises(ValueError): self.memory.add_plan(foreign)

    def test_normalized_csv_roundtrip_preserves_financial_inputs(self):
        from dataclasses import asdict
        original = load_holdings(CSV, 'holdings.csv')[0]
        restored = load_holdings(portfolio_csv(asdict(original)).encode(), 'saved.csv')[0]
        self.assertEqual(original.holdings, restored.holdings)
        self.assertEqual(original.goal, restored.goal)
        self.assertEqual(original.as_of_date, restored.as_of_date)


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.provider = Mock()
        self.provider.get_history.side_effect = AlphaVantageError('Unavailable')
        self.provider.get_quote.side_effect = AlphaVantageError('Unavailable')
        self.collection = Mock()
        self.collection.query.return_value = {
            'documents': [['Diversification reduces concentration risk.']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 2, 'category': 'investing'}]],
            'distances': [[0.2]],
        }
        self.explainer = Mock(model='test-model')
        self.explainer.generate.side_effect = RuntimeError('Test explanation unavailable')
        self.chat_generator = Mock(model='test-chat')
        self.chat_generator.generate.return_value = 'Diversification reduces concentration risk [1].'

        def service():
            store = ArtifactStore(persist=False)
            generators = {name: lambda: self.explainer for name in ['portfolio', 'market', 'goals']}
            generators['chat'] = lambda: self.chat_generator
            return WorkflowService(store=store, adapters=AgentAdapters(store, provider=self.provider,
                collection_factory=lambda: self.collection, generators=generators))

        self.memory = SessionMemory(profiles=ProfileStore(Path(self.tmp.name) / 'users'), service_factory=service)
        self.memory.profile.use_llm = False
        self.memory.profile.use_rag = False
        self.app = AppTest.from_file(str(ROOT / 'streamlit_app.py'))
        self.app.session_state['_memory'] = self.memory
        self.app.run(timeout=20)

    def click(self, label):
        next(b for b in self.app.button if b.label == label).click().run(timeout=20)
        self.assertEqual(len(self.app.exception), 0, [e.message for e in self.app.exception])

    def test_initial_tabs_and_reruns_do_not_call_providers(self):
        self.assertEqual([t.label for t in self.app.tabs], ['Chat', 'Portfolio', 'Market', 'Goals'])
        thread = self.memory.thread_id
        self.app.run(timeout=20)
        self.assertEqual(thread, self.memory.thread_id)
        self.provider.get_history.assert_not_called()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.toggle), 0)
        self.assertFalse(any('currency' in entry.label.lower() for entry in self.app.text_input))

    def test_example_portfolio_and_allocation_chart(self):
        self.click('Load example portfolios')
        self.assertEqual(len(self.app.selectbox(key='work_portfolio_choice').options), 15)
        self.click('Analyze portfolio')
        self.assertEqual(self.memory.latest['portfolio']['status'], 'complete')
        self.assertEqual(self.memory.latest['portfolio']['metrics']['narrative_calls'], 1)
        self.assertEqual(self.memory.latest['portfolio']['metrics']['retrieval_calls'], 1)
        self.assertNotIn('Run details', [e.label for e in self.app.expander])
        labels = [e.proto.label for e in self.app.get('download_button')]
        self.assertEqual(labels, ['Download CSV template'])
        self.assertFalse(any('Full report' in e.label for e in self.app.expander))
        self.assertIn('Portfolio value · USD', [m.label for m in self.app.metric])
        count = len(self.memory.messages)
        self.app.run(timeout=20)
        self.assertEqual(len(self.memory.messages), count)

    def test_goal_example_projection_and_selected_version(self):
        self.app.session_state['work_tabs'] = 'Goals'
        self.app.run(timeout=20)
        self.click('Load example goals')
        self.assertEqual(self.app.session_state['work_tabs'], 'Goals')
        self.click('Analyze selected goal')
        self.assertEqual(self.memory.latest['goals']['status'], 'complete')
        self.assertIn('47,400', [m.value for m in self.app.metric])
        self.assertEqual(self.app.selectbox(key='work_plan_choice').value, self.memory.selected_plan)
        count = len(self.memory.service.store.list(self.memory.thread_id))
        self.app.run(timeout=20)
        self.assertEqual(len(self.memory.service.store.list(self.memory.thread_id)), count)

    def test_goal_form_validates_and_calculates(self):
        self.app.text_input(key='work_goal_name').set_value('House deposit')
        self.app.date_input(key='work_goal_asof').set_value(date(2026, 11, 1))
        self.app.date_input(key='work_goal_deadline').set_value(date(2029, 11, 1))
        for key, value in [('target', 60000), ('savings', 15000), ('budget', 900), ('initial', 15000),
                           ('contribution', 900), ('return', 0), ('inflation', 0)]:
            self.app.number_input(key='work_goal_' + key).set_value(float(value))
        self.click('Add goal')
        self.assertNotIn('goals', self.memory.latest)
        self.click('Analyze selected goal')
        self.assertIn('47,400', [m.value for m in self.app.metric])
        self.provider.get_history.assert_not_called()

    def test_retirement_example_uses_explicit_long_term_assumptions(self):
        self.app.selectbox(key='work_example_goal').set_value('Retirement').run(timeout=20)
        self.click('Load example goals')
        plan = self.memory.service.store.get(self.memory.thread_id, self.memory.selected_plan)
        goal = plan['goals'][0]
        self.assertEqual(goal['id'], 'retirement')
        self.assertEqual(goal['target_basis'], 'today_dollars')
        self.assertEqual(goal['annual_return_pct'], 5)
        self.click('Analyze selected goal')
        response = self.memory.latest['goals']
        self.assertEqual(response['status'], 'complete')
        report = self.memory.service.get_report(self.memory.thread_id, response['result']['report_id'])['report']
        projection = report['results'][0]['projection']
        self.assertEqual(projection['contribution_months'], 360)
        self.assertGreater(projection['future_target'], goal['target_amount'])
        self.assertEqual(report['plan']['currency'], 'USD')

    def test_individual_goals_have_separate_analyses_and_results(self):
        self.click('Load example goals')
        house = self.memory.selected_plan
        self.app.selectbox(key='work_example_goal').set_value('Retirement').run(timeout=20)
        self.click('Load example goals')
        retirement = self.memory.selected_plan
        self.assertEqual(len(self.app.selectbox(key='work_plan_choice').options), 2)
        self.assertNotEqual(house, retirement)
        self.explainer.generate.assert_not_called()
        self.click('Analyze selected goal')
        saved = self.memory.service.get_report(self.memory.thread_id, self.memory.latest['goals']['result']['report_id'])
        self.assertEqual([g['name'] for g in saved['report']['plan']['goals']], ['Retirement nest egg'])
        self.app.selectbox(key='work_plan_choice').set_value(house).run(timeout=20)
        self.assertEqual(len(self.app.metric), 0)
        self.click('Analyze selected goal')
        self.assertIn('47,400', [m.value for m in self.app.metric])
        self.assertEqual(len(self.app.selectbox(key='work_plan_choice').options), 2)
        self.app.selectbox(key='work_plan_choice').set_value(retirement).run(timeout=20)
        self.assertIn('979,121', [m.value for m in self.app.metric])
        self.assertEqual(self.explainer.generate.call_count, 2)
        self.assertFalse(any('Review loaded' in e.label for e in self.app.expander))
        self.assertFalse(any('JSON' in str(e.proto.label) for e in self.app.get('file_uploader')))

    def test_incomplete_goal_form_pauses_without_assuming_returns(self):
        self.click('Add goal')
        self.assertTrue(len(self.app.warning))
        self.assertIn('Please check your inputs', self.app.warning[0].value)
        self.assertNotIn('annual_return_pct', self.app.warning[0].value)
        self.assertNotIn('goals', self.memory.latest)
        self.collection.query.assert_not_called()

    def test_chat_never_dispatches_market_or_portfolio_requests(self):
        self.app.chat_input[0].set_value('Analyze market').run(timeout=20)
        response = self.memory.messages[-1]['response']
        self.assertEqual(response['agent'], 'chat')
        self.assertEqual(response['metrics']['retrieval_calls'], 1)
        self.assertEqual(response['metrics']['narrative_calls'], 1)
        self.assertEqual(response['metrics']['router_calls'], 0)
        self.app.chat_input[0].set_value('Analyze holdings').run(timeout=20)
        self.assertEqual(self.memory.messages[-1]['response']['agent'], 'chat')
        self.provider.get_history.assert_not_called()
        self.assertEqual(self.memory.latest, {})

    def test_chat_cannot_see_analysis_inputs_reports_or_pending_requests(self):
        self.click('Load example goals')
        self.click('Analyze selected goal')
        self.assertEqual(self.memory.messages, [])
        self.memory.service.submit(self.memory.thread_id, {'message': 'Analyze market', 'use_llm': False, 'use_rag': False})
        self.app.chat_input[0].set_value('What is diversification?').run(timeout=20)
        self.assertEqual(self.memory.messages[-1]['response']['agent'], 'chat')
        prompt = self.chat_generator.generate.call_args.kwargs['prompt']
        self.assertNotIn('House deposit', prompt)
        self.assertNotIn('47,400', prompt)
        self.assertNotIn('60000', prompt)
        self.assertIsNotNone(self.memory.service.get_state(self.memory.thread_id)['pending_request'])
        self.assertNotEqual(self.memory.thread_id, self.memory.chat_thread_id)
        self.assertTrue(all(m['kind'] == 'report' for m in self.memory.chat_service.store.list(self.memory.chat_thread_id)))

    def test_new_conversation_clears_ui_drafts_and_reports(self):
        self.click('Load example goals')
        self.click('Analyze selected goal')
        old = self.memory.thread_id
        self.click('New conversation')
        self.assertNotEqual(old, self.memory.thread_id)
        self.assertEqual(self.memory.messages, [])
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(self.app.text_input(key='work_goal_name').value, '')

    def test_inline_disclaimer_is_removed_and_full_footer_is_present(self):
        from agents.chat.agent import DISCLAIMER
        from ui.render import response_text
        self.assertEqual(response_text('Answer.\n\nDisclaimer: ' + DISCLAIMER), 'Answer.')
        self.app.chat_input[0].set_value('What is diversification?').run(timeout=20)
        self.assertFalse(any(DISCLAIMER in element.value for element in self.app.markdown))
        self.assertTrue(any('not individualized investment, tax, or legal advice' in c.value for c in self.app.caption))

    def test_profile_creation_and_switch_isolate_widgets(self):
        self.app.text_input[0]  # Ensure the full app was rendered before form actions.
        name = next(t for t in self.app.text_input if t.label == 'Profile name')
        name.set_value('Alice')
        self.click('Create profile')
        self.assertEqual(self.memory.profile.name, 'Alice')
        self.app.text_input(key='work_goal_name').set_value('Alice goal').run(timeout=20)
        self.app.selectbox(key='profile_choice').set_value('guest').run(timeout=20)
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.text_input(key='work_goal_name').value, '')
        self.assertEqual(self.memory.profile_key, 'guest')

    def test_market_chart_uses_only_observed_analysis_window(self):
        days = ('2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30')
        bars = tuple(DailyBar(d, p, 1000) for d, p in zip(days, [100, 110, 99, 88, 105]))
        self.provider.get_history.side_effect = None
        self.provider.get_history.return_value = DailyHistory('AAPL', bars, 'raw_close', '2026-11-01T00:00:00+00:00')
        self.app.text_input(key='work_market_symbol').set_value('AAPL')
        self.app.slider(key='work_market_sessions').set_value(2)
        self.click('Analyze market')
        response = self.memory.latest['market']
        report = self.memory.service.get_report(self.memory.thread_id, response['result']['report_id'])['report']
        self.assertEqual(len(report['observed_prices']), 3)
        self.assertEqual(report['observed_prices'][0]['date'], report['metrics']['start_date'])
        self.assertEqual(report['observed_prices'][-1]['date'], report['metrics']['end_date'])
        self.provider.get_history.assert_called_once()


if __name__ == '__main__':
    unittest.main()
