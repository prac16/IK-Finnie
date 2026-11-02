import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.portfolio.analysis import PortfolioAnalysisAgent
from tools.openai_explanation import ExplanationDraft, DraftFinding, OpenAIExplanationGenerator
from agents.portfolio.explanation import (
    SYSTEM_PROMPT, explanation_context, explain_report,
)
from agents.portfolio.report import GroundedPortfolioReport, KnowledgeSection, build_report
from rag.retrieval import retrieve


def draft(**changes):
    data = dict(summary='The portfolio has concentrated exposure relative to its growth goal.',
                findings=[DraftFinding(title='Company concentration', observation='NVDA represents 30% of the portfolio.',
                                       goal_relevance='A decline in this company could affect the growth goal.',
                                       options=['Compare broader exposure with the current concentration.'],
                                       metric_fields=['top_holding_pct'], evidence_refs=[1])], limitations=[])
    data.update(changes)
    return ExplanationDraft(**data)


class LLMExplanationTests(unittest.TestCase):
    def setUp(self):
        self.portfolio = PortfolioAnalysisAgent().portfolios[3]
        self.collection = Mock()
        self.collection.query.return_value = {
            'documents': [['Diversification can reduce concentration risk.']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 2, 'category': 'investing'}]],
            'distances': [[.2]],
        }
        self.generator = Mock(model='test-model')
        self.generator.generate.return_value = draft()

    def test_one_llm_call_receives_report_goal_and_chunks_after_retrieval(self):
        def generate(context, **kwargs):
            self.collection.query.assert_called_once()
            self.assertEqual(context['goal'], 'Growth')
            self.assertEqual(context['report']['calculated_value_usd'], 320000)
            self.assertEqual(context['report']['top_holding_pct'], 30)
            self.assertEqual(context['retrieved_chunks'][0]['id'], 1)
            self.assertIn('Diversification', context['retrieved_chunks'][0]['text'])
            self.assertEqual(context['holdings'][0]['ticker'], 'NVDA')
            self.assertNotIn('input_source', json.dumps(context))
            self.assertNotIn('analysis_focus', json.dumps(context))
            return draft()
        self.generator.generate.side_effect = generate
        result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
        self.generator.generate.assert_called_once()
        self.assertEqual(result.explanation.model, 'test-model')
        self.assertIsNone(result.explanation.fallback_reason)
        self.assertEqual(result.analysis.calculated_value_usd, 320000)
        self.assertIn('LLM explanation', result.to_markdown())
        self.assertIn('test-model', result.to_markdown())

    def test_invalid_citations_metric_names_numbers_use_labeled_fallback(self):
        for field, value in (('evidence_refs', [99]), ('metric_fields', ['invented_sharpe']),
                             ('observation', 'NVDA represents 999% of the portfolio.')):
            with self.subTest(field=field):
                bad = draft()
                bad.findings[0] = bad.findings[0].model_copy(update={field: value})
                self.generator.generate.return_value = bad
                result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
                self.assertEqual(result.explanation.method, 'Local fallback; not LLM-generated')
                self.assertIsNotNone(result.explanation.fallback_reason)
                self.assertEqual(result.analysis.top_holding_pct, 30)

    def test_unestablished_goal_alignment_claim_is_rejected(self):
        self.generator.generate.return_value = draft(summary='This portfolio aligns with your growth objective.')
        result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
        self.assertEqual(result.explanation.method, 'Local fallback; not LLM-generated')
        self.assertIsNotNone(result.explanation.fallback_reason)
        self.assertEqual(result.analysis.calculated_value_usd, 320000)

    def test_conditional_goal_option_is_not_mistaken_for_suitability_claim(self):
        good = draft()
        good.findings[0].options = ['Compare broader exposure while ensuring choices align with the stated growth goal.']
        self.generator.generate.return_value = good
        result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
        self.assertIsNone(result.explanation.fallback_reason)

    def test_absent_crypto_discussion_is_rejected_for_growth_portfolio(self):
        bad = draft(summary='Missing crypto means missed opportunities.')
        self.generator.generate.return_value = bad
        result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
        self.assertEqual(result.explanation.method, 'Local fallback; not LLM-generated')
        self.assertNotIn('crypto_pct', self.generator.generate.call_args.args[0]['report'])

    def test_api_failure_preserves_report_and_does_not_expose_error(self):
        self.generator.generate.side_effect = RuntimeError('secret-provider-error')
        result = build_report(self.portfolio, self.collection, explanation_generator=self.generator)
        self.assertEqual(result.analysis.calculated_value_usd, 320000)
        self.assertTrue(result.knowledge)
        self.assertIn('fallback', result.to_markdown())
        self.assertNotIn('secret-provider-error', result.to_markdown())
        self.generator.generate.assert_called_once()

    def test_missing_goal_prohibits_actions_and_no_sources_prohibits_citations(self):
        p = replace(self.portfolio, goal='Not supplied')
        result = build_report(p, explanation_generator=self.generator)
        self.assertIsNone(result.explanation.goal)
        self.assertIsNotNone(result.explanation.fallback_reason)
        self.assertTrue(all(not f.options for f in result.explanation.findings))

    def test_disabled_llm_does_not_construct_client(self):
        with patch('tools.openai_explanation.OpenAI') as client:
            result = build_report(self.portfolio, self.collection, use_llm=False)
        client.assert_not_called()
        self.assertIsNone(result.explanation.model)

    @patch('tools.openai_explanation.load_dotenv')
    @patch('tools.openai_explanation.OpenAI')
    def test_sdk_uses_strict_schema_one_request_no_retries(self, constructor, dotenv):
        constructor.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=draft().model_dump_json(), refusal=None))])
        generator = OpenAIExplanationGenerator(model='test-model')
        parsed = generator.generate({'goal': 'Growth'}, system=SYSTEM_PROMPT)
        self.assertEqual(parsed.findings[0].evidence_refs, [1])
        constructor.assert_called_once_with(timeout=45.0, max_retries=0)
        request = constructor.return_value.chat.completions.create
        request.assert_called_once()
        kwargs = request.call_args.kwargs
        self.assertFalse(kwargs['store'])
        self.assertTrue(kwargs['response_format']['json_schema']['strict'])
        self.assertEqual(kwargs['messages'][0]['content'], SYSTEM_PROMPT)
        self.assertIn('untrusted', SYSTEM_PROMPT)
        self.assertIn('do not expand', SYSTEM_PROMPT.lower())

    @patch('tools.openai_explanation.load_dotenv')
    @patch('tools.openai_explanation.OpenAI')
    def test_refused_response_falls_back(self, constructor, dotenv):
        constructor.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=None, refusal='Refused'))])
        result = build_report(self.portfolio, self.collection)
        self.assertIsNotNone(result.explanation.fallback_reason)
