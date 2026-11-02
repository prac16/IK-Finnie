from functools import partial
import io
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from openpyxl import Workbook

from agents.portfolio.analysis import PortfolioAnalysisAgent, DEFAULT_SAMPLE_WORKBOOK
from agents.portfolio.holdings import load_holdings
from agents.portfolio.report import build_report as _build_report
build_report = partial(_build_report, use_llm=False)
from agents.portfolio.upload import make_handler

HEADER = 'ticker,asset_type,market_value_usd,portfolio,as_of_date\n'


def csv_file(rows):
    return (HEADER + rows).encode()


class UploadTests(unittest.TestCase):
    def test_all_15_samples_reconcile_against_uploaded_xlsx_and_csv(self):
        baseline = PortfolioAnalysisAgent().analyze_all()
        xlsx = load_holdings(DEFAULT_SAMPLE_WORKBOOK.read_bytes(), 'samples.xlsx')
        csv = load_holdings(DEFAULT_SAMPLE_WORKBOOK.with_name('sample_portfolio_holdings.csv').read_bytes(), 'samples.csv')
        self.assertEqual(len(xlsx), 15)
        self.assertEqual(len(csv), 15)
        for expected, x, c in zip(baseline, xlsx, csv):
            for portfolio in (x, c):
                report = build_report(portfolio).analysis
                self.assertAlmostEqual(report.calculated_value_usd, expected.calculated_value_usd)
                self.assertEqual(report.asset_allocation_pct, expected.asset_allocation_pct)
                self.assertEqual(report.sector_allocation_pct, expected.sector_allocation_pct)
                self.assertAlmostEqual(report.hhi, expected.hhi)
                self.assertAlmostEqual(sum(h.weight_pct for h in report.portfolio.holdings), 100)

    def test_duplicate_tickers_combined_before_concentration(self):
        portfolio = load_holdings(csv_file('AAPL,Stock,30,P,2026-10-30\nAAPL,Stock,30,P,2026-10-30\nBND,Bond ETF,40,P,2026-10-30\n'), 'p.csv')[0]
        report = build_report(portfolio).analysis
        self.assertEqual(len(report.portfolio.holdings), 2)
        self.assertEqual(report.top_holding_pct, 60)
        self.assertAlmostEqual(report.hhi, .52)
        self.assertAlmostEqual(report.effective_holdings, 1/.52)
        self.assertEqual(report.portfolio.as_of_date, '2026-10-30')

    def test_invalid_data_rejected(self):
        for rows in ('AAPL,Stock,nan,P,\n', 'AAPL,Stock,inf,P,\n', 'AAPL,Stock,-1,P,\n',
                     'AAPL,Stock,0,P,\n', ',Stock,100,P,\n', 'AAPL,Unknown,100,P,\n',
                     'AAPL,Stock,100,P,wrong\n', 'AAPL,Stock,10,P,2026-10-30\nBND,Bond ETF,10,P,2026-10-29\n'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_holdings(csv_file(rows), 'p.csv')

    def test_missing_columns_bad_weights_and_non_usd_rejected(self):
        for content in (b'ticker\nAAPL\n', b'ticker,asset_type,market_value_usd,weight_pct\nAAPL,Stock,100,40\n',
                        b'ticker,asset_type,market_value_usd,currency\nAAPL,Stock,100,EUR\n',
                        b'ticker,ticker,asset_type,market_value_usd\nAAPL,AAPL,Stock,100\n'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_holdings(content, 'p.csv')

    def test_explicit_class_supports_non_us_stock(self):
        p = load_holdings(b'ticker,asset_type,asset_class,market_value_usd\nASML,Stock,International equity,100\n', 'p.csv')[0]
        self.assertEqual(build_report(p).analysis.asset_allocation_pct, {'International equity': 100})

    def test_excel_upload_and_formula_rejection(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(['ticker', 'asset_type', 'market_value_usd'])
        sheet.append(['BND', 'Bond ETF', 100])
        buffer = io.BytesIO()
        workbook.save(buffer)
        self.assertEqual(load_holdings(buffer.getvalue(), 'p.xlsx')[0].stated_value_usd, 100)
        sheet['C2'] = '=50+50'
        buffer = io.BytesIO()
        workbook.save(buffer)
        with self.assertRaisesRegex(ValueError, 'formulas'):
            load_holdings(buffer.getvalue(), 'p.xlsx')
        with self.assertRaises(ValueError):
            load_holdings(b'bad workbook', 'p.xlsx')

    def test_rag_uses_ranked_excerpts_and_metrics_not_quote_tool(self):
        collection = Mock()
        collection.query.return_value = {
            'documents': [['Diversification reduces concentration.']],
            'metadatas': [[{'source': 'Risk.pdf', 'page': 2, 'category': 'investing'}]],
            'distances': [[0.2]],
        }
        p = load_holdings(csv_file('AAPL,Stock,60,P,\nBND,Bond ETF,40,P,\n'), 'p.csv')[0]
        with patch('tools.alpha_vantage.AlphaVantageTickerTool', side_effect=AssertionError('Market tool forbidden')):
            report = build_report(p, collection)
        self.assertEqual(report.analysis.top_holding_pct, 60)
        self.assertIn('Diversification reduces concentration.', report.to_markdown())
        self.assertIn('Risk.pdf', report.to_markdown())
        self.assertEqual(report.knowledge[0].retrieval.metrics.selected_count, 1)
        self.assertEqual(collection.query.call_args.kwargs['where'], {'category': 'investing'})
        json.dumps(report.to_dict(), allow_nan=False)

    def test_rag_failure_preserves_calculations_and_warns(self):
        collection = Mock()
        collection.query.side_effect = RuntimeError('index unavailable')
        p = load_holdings(csv_file('AAPL,Stock,100,P,\n'), 'p.csv')[0]
        report = build_report(p, collection)
        self.assertEqual(report.analysis.calculated_value_usd, 100)
        self.assertIn('Retrieval unavailable', report.warnings[0])
        self.assertIsNone(report.analysis.portfolio.age)
        self.assertIn('not supplied', report.to_markdown())

    def test_existing_engine_rejects_nonfinite_and_inconsistent_weights(self):
        p = PortfolioAnalysisAgent().portfolios[0]
        for bad in (replace(p.holdings[0], market_value_usd=float('nan')),
                    replace(p.holdings[0], weight_pct=p.holdings[0].weight_pct-1)):
            with self.assertRaises(ValueError):
                PortfolioAnalysisAgent._analyze_portfolio(replace(p, holdings=(bad,) + p.holdings[1:]))

    def test_http_upload_runs_shared_pipeline_without_socket(self):
        handler = object.__new__(make_handler('secret'))
        content = csv_file('AAPL,Stock,60,P,\nBND,Bond ETF,40,P,\n')
        handler.headers = {'X-Upload-Token': 'secret', 'Content-Length': str(len(content))}
        handler.path = '/analyze?filename=test.csv&rag=false&llm=false'
        handler.rfile = io.BytesIO(content)
        handler.send = Mock()
        handler.do_POST()
        status, body, _ = handler.send.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['reports'][0]['analysis']['top_holding_pct'], 60)
        handler.headers['X-Upload-Token'] = 'wrong'
        handler.do_POST()
        self.assertEqual(handler.send.call_args.args[0], 403)

    def test_tech_heavy_sector_allocation_and_coverage(self):
        report = PortfolioAnalysisAgent().analyze("Tech Heavy")
        self.assertEqual(report.sector_allocation_pct, {
            "Information Technology": 75, "Communication Services": 15, "Not applicable": 10,
        })
        self.assertEqual(report.known_technology_stock_pct, 75)
        self.assertEqual(report.sector_coverage_pct, 100)
        self.assertTrue(any("three largest companies total 75.0%" in f for f in report.findings))

    def test_missing_sector_does_not_guess_from_ticker(self):
        portfolio = load_holdings(csv_file('AAPL,Stock,100,P,\n'), 'p.csv')[0]
        report = build_report(portfolio).analysis
        self.assertEqual(report.sector_allocation_pct, {"Unknown": 100})
        self.assertEqual(report.sector_coverage_pct, 0)
        self.assertEqual(report.known_technology_stock_pct, 0)

    def test_broad_funds_are_not_counted_as_sector_coverage(self):
        report = PortfolioAnalysisAgent().analyze('ETF Only')
        self.assertEqual(report.sector_allocation_pct, {"Multi-sector": 100})
        self.assertEqual(report.sector_coverage_pct, 0)
        for report in PortfolioAnalysisAgent().analyze_all():
            self.assertAlmostEqual(sum(report.sector_allocation_pct.values()), 100)
            self.assertTrue(all(h.sector and h.sector_source for h in report.portfolio.holdings))

    def test_invalid_or_conflicting_sectors_are_rejected(self):
        for rows in ('AAPL,Stock,100,Technology\n', 'BND,Bond ETF,100,Information Technology\n',
                     'AAPL,Stock,100,Not applicable\n', 'AAPL,Stock,100,Multi-sector\n'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_holdings(('ticker,asset_type,market_value_usd,sector\n'+rows).encode(), 'p.csv')
        p = load_holdings(b'ticker,asset_type,market_value_usd,sector\nAAPL,Stock,50,Information Technology\nAAPL,Stock,50,Financials\n', 'p.csv')[0]
        with self.assertRaisesRegex(ValueError, 'Conflicting classifications'):
            build_report(p)
