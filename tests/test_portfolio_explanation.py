from functools import partial
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from agents.portfolio.analysis import PortfolioAnalysisAgent
from agents.portfolio.explanation import explain_report
from agents.portfolio.report import build_report as _build_report
from rag.portfolio import plan_retrieval
build_report = partial(_build_report, use_llm=False)


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = PortfolioAnalysisAgent()

    def plan(self, portfolio):
        return plan_retrieval(self.agent._analyze_portfolio(portfolio))

    def test_retrieval_plan_uses_goal_and_holdings_not_name_or_focus(self):
        p = self.agent.portfolios[3]
        misleading = replace(p, name='Retirement Tax Insurance Estate Planning',
                             analysis_focus='Retirement tax college insurance')
        self.assertEqual(self.plan(p), self.plan(misleading))
        self.assertEqual(list(self.plan(misleading).topics), ['Allocation and concentration'])
        actual_tax_goal = replace(p, goal='Tax-efficient growth')
        self.assertIn('Tax considerations', self.plan(actual_tax_goal).topics)

    def test_findings_ignore_unrelated_name_and_analysis_focus(self):
        p = self.agent.portfolios[3]
        base = self.agent._analyze_portfolio(p)
        changed = self.agent._analyze_portfolio(replace(p, name='Inflation Hedge Dividend Tax Retirement', analysis_focus='Tax income and preservation dividend'))
        self.assertEqual(base.findings, changed.findings)
        self.assertEqual(base.suggestions, changed.suggestions)
        self.assertEqual(base.data_limitations, changed.data_limitations)

    def test_goal_related_topics_require_relevant_holdings_where_needed(self):
        p = self.agent.portfolios[3]
        self.assertNotIn('Speculative exposure', self.plan(p).topics)
        crypto = next(p for p in self.agent.portfolios if p.name == 'Crypto Heavy')
        self.assertIn('Speculative exposure', self.plan(crypto).topics)
        inflation = next(p for p in self.agent.portfolios if p.name == 'Inflation Hedge')
        self.assertIn('Purchasing power', self.plan(inflation).topics)

    def test_one_retrieval_occurs_after_metrics_and_before_explanation(self):
        p = self.agent.portfolios[3]
        events = []
        analyze = self.agent._analyze_portfolio
        def calculate(portfolio):
            result = analyze(portfolio)
            events.append('metrics complete')
            return result
        def query(**kwargs):
            self.assertEqual(events, ['metrics complete'])
            self.assertIn('largest NVDA 30.0%', kwargs['query_texts'][0])
            events.append('retrieval')
            return {'documents': [[]]}
        def explain(report, **kwargs):
            self.assertEqual(events, ['metrics complete', 'retrieval'])
            events.append('explanation')
            return explain_report(report, use_llm=False)
        collection = Mock()
        collection.query.side_effect = query
        with patch.object(PortfolioAnalysisAgent, '_analyze_portfolio', side_effect=calculate), patch('agents.portfolio.report.explain_report', side_effect=explain):
            report = build_report(p, collection)
        collection.query.assert_called_once()
        self.assertEqual(events, ['metrics complete', 'retrieval', 'explanation'])
        self.assertEqual(report.analysis.calculated_value_usd, 320000)

    def test_all_samples_use_one_query_even_with_multiple_topics(self):
        for p in self.agent.portfolios:
            with self.subTest(portfolio=p.name):
                collection = Mock()
                collection.query.return_value = {'documents': [[]]}
                build_report(p, collection)
                collection.query.assert_called_once()
                self.assertEqual(collection.query.call_args.kwargs['n_results'], 20)

    def test_explanation_consumes_completed_report_without_recalculating(self):
        report = build_report(self.agent.portfolios[3])
        altered = replace(report, analysis=replace(report.analysis, calculated_value_usd=123456, top_holding_pct=42.3))
        with patch.object(PortfolioAnalysisAgent, '_analyze_portfolio', side_effect=AssertionError('Must not recalculate')):
            text = explain_report(altered, use_llm=False).to_markdown()
        self.assertIn('$123,456.00', text)
        self.assertIn('42.3%', text)
        self.assertIn('Growth', text)
        self.assertNotIn('insurance', text.lower())
        self.assertNotIn('estate planning', text.lower())
        self.assertEqual(report.analysis.calculated_value_usd, 320000)

    def test_explanation_is_after_metrics_and_evidence_and_preserves_reference_ids(self):
        collection = Mock()
        collection.query.return_value = {
            'documents': [['Diversification context']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 2, 'category': 'investing'}]],
            'distances': [[.2]],
        }
        def explain(completed, **kwargs):
            self.assertIsNone(completed.explanation)
            self.assertEqual(completed.analysis.calculated_value_usd, 320000)
            self.assertEqual(completed.knowledge[0].retrieval.chunks[0].source, 'Risk.pdf')
            return explain_report(completed, use_llm=False)
        with patch('agents.portfolio.report.explain_report', side_effect=explain) as explainer:
            report = build_report(self.agent.portfolios[3], collection)
        explainer.assert_called_once()
        self.assertEqual(report.explanation.findings[0].evidence_refs, (1,))
        self.assertIn('explanation', report.to_dict())
        self.assertIn('What this means', report.to_markdown())

    def test_missing_goal_has_no_goal_specific_actions(self):
        p = replace(self.agent.portfolios[3], goal='Not supplied')
        report = build_report(p)
        self.assertIsNone(report.explanation.goal)
        self.assertTrue(all(not f.options for f in report.explanation.findings))
        self.assertFalse(report.analysis.suggestions)
        self.assertIn('No goal was supplied', report.explanation.to_markdown())
