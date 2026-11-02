"""Assert privacy on actual SDK request messages, without transmitting test PII."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from agents.goals.agent import GoalsPlanningAgent
from agents.goals.explanation import explanation_context
from tools.llm_privacy import PrivatePayload, privacy_scope, redact_text
from tools.openai_explanation import OpenAIExplanationGenerator
from tools.openai_generation import OpenAIGenerator
from tools.openai_goals import OpenAIGoalsExplanationGenerator
from tools.openai_market import OpenAIMarketExplanationGenerator
from tools.openai_router import OpenAIRouter
from ui.controller import SessionMemory
from ui.memory import ProfileStore
from workflow import WorkflowService
from workflow.adapters import AgentAdapters
from workflow.artifacts import ArtifactStore

CANARIES = ['Jane Doe', 'jane@example.com', '212-555-0198', '123-45-6789', '987654321099',
            '42 Willow Street', '1980-03-09']
PII = 'Jane Doe jane@example.com 212-555-0198 SSN 123-45-6789 account 987654321099 42 Willow Street DOB 1980-03-09'


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
        content=json.dumps(content) if not isinstance(content, str) else content, refusal=None))])


def goal_plan():
    plan = json.loads((Path(__file__).resolve().parents[1] / 'src/data/sample_goals_plan.json').read_text())
    plan['goals'][0].update(id='jane_house', name='House deposit for Jane Doe')
    return plan


class PayloadTests(unittest.TestCase):
    def assert_private(self, payload):
        text = json.dumps(payload)
        for canary in CANARIES:
            self.assertNotIn(canary, text)

    def test_report_allowlist_and_goal_purpose_preserve_numbers(self):
        context = {'goal': 'Growth for ' + PII, 'owner_name': 'Jane Doe', 'age': 46,
                   'input_source': '/Users/jane/accounts.csv', 'portfolio_id': 'jane_private',
                   'holdings': [{'ticker': 'AAPL', 'weight_pct': 60, 'market_value_usd': 6000, 'sector_source': PII}],
                   'report': {'calculated_value_usd': 10000, 'top_holding_pct': 60, 'owner': PII, 'findings': [PII]},
                   'retrieved_chunks': [{'id': 1, 'source': 'jane_private_statement.pdf', 'page': 2, 'text': PII}],
                   'user_question': 'Explain concentration for ' + PII}
        original = deepcopy(context)
        payload = PrivatePayload(context, 'portfolio').data
        self.assert_private(payload)
        self.assertNotIn('jane_private', json.dumps(payload))
        self.assertEqual(payload['goal'], 'growth')
        self.assertEqual(payload['holdings'][0]['market_value_usd'], 6000)
        self.assertEqual(payload['report']['calculated_value_usd'], 10000)
        self.assertEqual(payload['retrieved_chunks'][0]['id'], 1)
        self.assertEqual(payload['retrieved_chunks'][0]['page'], 2)
        self.assertEqual(context, original)

    def test_goal_aliases_are_reversible_and_paths_stay_valid(self):
        report = GoalsPlanningAgent(use_llm=False).plan(goal_plan())
        context = explanation_context(report)
        payload = PrivatePayload(context, 'goals')
        serialized = json.dumps(payload.data)
        self.assertNotIn('jane_house', serialized)
        self.assertNotIn('Jane Doe', serialized)
        self.assertIn('results.goal_a.projection.shortfall', serialized)
        self.assertEqual(payload.data['results']['goal_a']['goal']['id'], 'goal_a')
        self.assertEqual(payload.data['results']['goal_a']['projection']['ending_balance'], 47400)
        restored = payload.restore({'goal_id': 'goal_a', 'field_paths': ['results.goal_a.projection.shortfall']})
        self.assertEqual(restored['goal_id'], 'jane_house')
        self.assertEqual(restored['field_paths'], ['results.jane_house.projection.shortfall'])

    def test_goal_name_equal_to_id_does_not_break_aliases(self):
        plan = goal_plan()
        plan['goals'][0]['name'] = plan['goals'][0]['id']
        data = PrivatePayload(explanation_context(GoalsPlanningAgent(use_llm=False).plan(plan)), 'goals').data
        self.assertEqual(data['results']['goal_a']['goal']['id'], 'goal_a')

    def test_router_excludes_history_labels_and_internal_ids(self):
        context = {'message': PII + ' can I reach my goal?', 'recent_messages': [{'content': PII}],
                   'active_context': {'goal_ids': ['jane_house'], 'portfolio_id': 'private_portfolio', 'plan_id': 'private_plan', 'last_agent': 'goals'},
                   'artifacts': [{'id': 'private_file', 'name': PII, 'kind': 'goal_plan'}]}
        payload = PrivatePayload(context, 'router')
        self.assert_private(payload.data)
        for value in ['jane_house', 'private_plan', 'private_portfolio', 'private_file', 'recent_messages']:
            self.assertNotIn(value, json.dumps(payload.data))
        self.assertEqual(payload.restore({'goal_id': 'goal_a'})['goal_id'], 'jane_house')

    def test_known_profile_names_are_call_scoped(self):
        with privacy_scope(['Alexandra']):
            self.assertNotIn('Alexandra', redact_text('Explain this for Alexandra.'))
        self.assertIn('Alexandra', redact_text('Explain this for Alexandra.'))

    def test_financial_values_and_dates_are_not_phone_numbers(self):
        self.assertEqual(redact_text('AAPL returned 3.28% on 2026-10-30; budget 1,250.00.'),
                         'AAPL returned 3.28% on 2026-10-30; budget 1,250.00.')


class TransportTests(unittest.TestCase):
    def assert_messages_private(self, client):
        kwargs = client.return_value.chat.completions.create.call_args.kwargs
        text = json.dumps(kwargs['messages'])
        for canary in CANARIES:
            self.assertNotIn(canary, text)
        self.assertFalse(kwargs['store'])
        return json.loads(kwargs['messages'][1]['content'])

    def test_portfolio_transport(self):
        with patch('tools.openai_explanation.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response({
                'summary': 'Measured concentration.', 'findings': [{'title': 'Exposure', 'observation': '60%',
                'goal_relevance': 'Consider the stated goal.', 'options': [], 'metric_fields': ['top_holding_pct'], 'evidence_refs': []}], 'limitations': []})
            generator = OpenAIExplanationGenerator()
            generator.generate({'goal': PII, 'report': {'top_holding_pct': 60}, 'holdings': [], 'user_question': PII}, system='Explain metrics.')
            self.assert_messages_private(client)

    def test_goals_transport_restores_ids_before_existing_agent_validation(self):
        with patch('tools.openai_goals.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response({
                'summary': 'A shortfall exists under the assumptions.', 'findings': [{
                    'goal_id': 'goal_a', 'observation': 'The shortfall is 12,600.',
                    'explanation': 'Contributions are insufficient at the stated deadline.', 'next_steps': [],
                    'recommended_scenarios': [], 'field_paths': ['results.goal_a.projection.shortfall'], 'reference_ids': []}], 'limitations': []})
            report = GoalsPlanningAgent(generator=OpenAIGoalsExplanationGenerator(model='privacy-test')).plan(goal_plan())
            data = self.assert_messages_private(client)
            self.assertNotIn('jane_house', json.dumps(data))
            self.assertEqual(report.explanation.model, 'privacy-test')
            self.assertEqual(report.explanation.findings[0]['goal_id'], 'jane_house')
            self.assertEqual(report.plan['goals'][0]['name'], 'House deposit for Jane Doe')

    def test_market_and_followup_transport(self):
        with patch('tools.openai_market.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response({
                'summary': 'Measured return.', 'findings': [{'title': 'Return', 'observation': '5%',
                    'interpretation': 'Historical only.', 'metric_fields': ['period_return_pct'], 'reference_ids': []}], 'limitations': []})
            OpenAIMarketExplanationGenerator().generate({'request': {'ticker': 'AAPL', 'owner_name': PII},
                'metrics': {'period_return_pct': 5}, 'user_question': PII + ' drawdown', 'warnings': [PII]}, system='Explain metrics.')
            data = self.assert_messages_private(client)
            self.assertEqual(data['metrics']['period_return_pct'], 5)

    def test_router_transport_restores_goal_selection(self):
        with patch('tools.openai_router.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response({
                'agent': 'goals', 'action': 'analyze', 'symbol': None, 'sessions': None, 'benchmark': None,
                'adjusted': None, 'goal_id': 'goal_a', 'monthly_contribution': 800, 'monthly_budget': None, 'question': None})
            result = OpenAIRouter().route({'message': PII, 'active_context': {'goal_ids': ['jane_house']},
                                           'artifacts': [], 'recent_messages': [{'content': PII}]})
            self.assert_messages_private(client)
            self.assertEqual(result.goal_id, 'jane_house')

    def test_chat_transport_filters_prompt_and_system(self):
        with patch('tools.openai_generation.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response('A grounded answer.')
            OpenAIGenerator().generate(system='Explain for ' + PII, prompt='Question: ' + PII)
            kwargs = client.return_value.chat.completions.create.call_args.kwargs
            for canary in CANARIES:
                self.assertNotIn(canary, json.dumps(kwargs['messages']))
            self.assertFalse(kwargs['store'])

    def test_profile_name_is_redacted_through_the_actual_chat_graph(self):
        collection = Mock()
        collection.query.return_value = {'documents': [['Diversification reduces risk.']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 1}]], 'distances': [[0.1]]}
        def service():
            store = ArtifactStore(persist=False)
            return WorkflowService(store=store, adapters=AgentAdapters(store, collection_factory=lambda: collection))
        with TemporaryDirectory() as directory, patch('tools.openai_generation.OpenAI') as client:
            client.return_value.chat.completions.create.return_value = response('Diversification reduces risk [1].')
            memory = SessionMemory(profiles=ProfileStore(Path(directory)), service_factory=service)
            memory.create_profile('Renata')
            answer = memory.ask_chat('Explain diversification to Renata.')
            self.assertEqual(answer.status, 'complete')
            messages = client.return_value.chat.completions.create.call_args.kwargs['messages']
            self.assertNotIn('Renata', json.dumps(messages))


if __name__ == '__main__':
    unittest.main()
