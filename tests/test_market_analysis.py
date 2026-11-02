import json
import math
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from agents.market.agent import MarketAnalysisAgent
from agents.market.analysis import calculate_metrics
from tools.market_data import AlphaVantageMarketData, DailyBar, DailyHistory, MarketDataUnavailable
from tools.openai_market import MarketExplanationDraft, MarketFindingDraft, OpenAIMarketExplanationGenerator


def history(symbol='AAPL', closes=(100, 110, 99, 88, 105), **kwargs):
    days = ('2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30')
    bars = tuple(DailyBar(day, value, (index+1)*100) for index, (day,value) in enumerate(zip(days, closes)))
    return DailyHistory(symbol, bars, 'raw_close', '2026-10-31T00:00:00+00:00', **kwargs)


def payload(symbol='AAPL', adjusted=False):
    rows={}
    for bar in history(symbol).bars:
        row={'4. close':str(bar.close), '5. volume':str(bar.volume)}
        if adjusted:
            row={'4. close':str(bar.close), '5. adjusted close':str(bar.close),
                 '6. volume':str(bar.volume), '8. split coefficient':'1'}
        rows[bar.date]=row
    return {'Meta Data':{'2. Symbol':symbol, '5. Time Zone':'US/Eastern'}, 'Time Series (Daily)':rows}


class Response:
    def __init__(self, data):
        self.data=json.dumps(data).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,limit): return self.data[:limit]


class MarketDataTests(unittest.TestCase):
    def test_usd_workspace_rejects_foreign_listing_suffixes(self):
        from tools.market_data import normalize_us_symbol
        for symbol in ['TSCO.LON', 'SHOP.TRT', '0700.HKG', 'RELIANCE.BSE']:
            with self.assertRaises(ValueError): normalize_us_symbol(symbol)
        self.assertEqual(normalize_us_symbol('brk.b'), 'BRK.B')

    def test_endpoint_normalization_sorting_and_cache(self):
        opener=Mock(return_value=Response(payload()))
        now=[10.0]
        tool=AlphaVantageMarketData(api_key='test',opener=opener,clock=lambda:now[0])
        first=tool.get_history(' aapl ')
        now[0]=15
        cached=tool.get_history('AAPL')
        params=parse_qs(urlparse(opener.call_args.args[0].full_url).query)
        self.assertEqual(params['function'],['TIME_SERIES_DAILY'])
        self.assertEqual(params['outputsize'],['compact'])
        self.assertEqual(first.bars[0].date,'2026-10-26')
        self.assertEqual(cached.source,'cache')
        self.assertEqual(cached.cache_age_seconds,5)
        opener.assert_called_once()

    def test_adjusted_is_explicit_and_never_silently_changes_basis(self):
        opener=Mock(return_value=Response(payload(adjusted=True)))
        tool=AlphaVantageMarketData(api_key='test',opener=opener)
        result=tool.get_history('AAPL',adjusted=True)
        self.assertEqual(result.price_basis,'adjusted_close')
        self.assertIn('TIME_SERIES_DAILY_ADJUSTED',opener.call_args.args[0].full_url)

    def test_two_exponential_jitter_retries_then_stale_cache_with_age_limit(self):
        now=[0.0]
        opener=Mock(side_effect=[Response(payload()),TimeoutError(),TimeoutError(),TimeoutError(),TimeoutError(),TimeoutError(),TimeoutError()])
        sleeper,jitter=Mock(),Mock(side_effect=[.5,1.5,.5,1.5])
        tool=AlphaVantageMarketData(api_key='test',opener=opener,clock=lambda:now[0],sleeper=sleeper,jitter=jitter,
                                   cache_ttl_seconds=1,max_stale_seconds=10)
        tool.get_history('AAPL')
        now[0]=5
        result=tool.get_history('AAPL')
        self.assertEqual(result.freshness,'stale')
        self.assertEqual(result.attempts,3)
        self.assertEqual([c.args for c in jitter.call_args_list],[(0,1.0),(0,2.0)])
        now[0]=11
        with self.assertRaises(MarketDataUnavailable): tool.get_history('AAPL')
        self.assertEqual(opener.call_count,7)

    def test_rate_limit_cooldown_skips_additional_requests(self):
        opener=Mock(return_value=Response({'Information':'25 requests per day rate limit reached'}))
        tool=AlphaVantageMarketData(api_key='test',opener=opener,sleeper=lambda _:None)
        with self.assertRaises(MarketDataUnavailable): tool.get_history('AAPL')
        with self.assertRaises(MarketDataUnavailable): tool.get_history('MSFT')
        self.assertEqual(opener.call_count,3)

    def test_invalid_and_nonfinite_provider_data_rejected(self):
        bads=[]
        wrong=payload('MSFT'); bads.append(wrong)
        for value in ('NaN','Infinity','-1','0'):
            data=payload();data['Time Series (Daily)']['2026-10-26']['4. close']=value;bads.append(data)
        data=payload();data['Time Series (Daily)']['bad-date']={'4. close':'1','5. volume':'10'};bads.append(data)
        for data in bads:
            with self.subTest(data=data):
                opener=Mock(return_value=Response(data))
                tool=AlphaVantageMarketData(api_key='test',opener=opener)
                with self.assertRaises(MarketDataUnavailable): tool.get_history('AAPL')
                opener.assert_called_once()

    def test_missing_key_and_bad_symbol_do_not_call_api(self):
        opener=Mock()
        tool=AlphaVantageMarketData(api_key='',opener=opener)
        with self.assertRaises(MarketDataUnavailable):tool.get_history('AAPL')
        with self.assertRaises(ValueError):tool.get_history('AAPL&apikey=secret')
        opener.assert_not_called()


class MarketMetricsTests(unittest.TestCase):
    def test_known_drawdown_return_and_volume(self):
        m,warnings=calculate_metrics(history(),4)
        self.assertAlmostEqual(m.period_return_pct,5)
        self.assertAlmostEqual(m.max_drawdown_pct,20)
        self.assertAlmostEqual(m.daily_change_pct,(105/88-1)*100)
        self.assertEqual(m.volume_vs_prior_average,2)
        self.assertEqual(m.return_intervals,4)
        self.assertFalse(warnings)

    def test_constant_returns_have_zero_volatility(self):
        m,_=calculate_metrics(history(closes=(100,110,121)),2)
        self.assertAlmostEqual(m.annualized_volatility_pct,0)
        self.assertAlmostEqual(m.period_return_pct,21)
        self.assertEqual(m.max_drawdown_pct,0)

    def test_same_date_benchmark_comparison_is_percentage_points(self):
        m,_=calculate_metrics(history(),4,history('SPY',(50,51,52,53,55)))
        self.assertAlmostEqual(m.benchmark_return_pct,10)
        self.assertAlmostEqual(m.benchmark_difference_pp,-5)

    def test_short_window_is_explicit_and_no_volatility_for_one_return(self):
        m,warnings=calculate_metrics(history(closes=(100,110)),20)
        self.assertEqual(m.return_intervals,1)
        self.assertIsNone(m.annualized_volatility_pct)
        self.assertTrue(warnings)

    def test_adjusted_returns_use_adjusted_close(self):
        raw=history(closes=(100,50))
        adjusted=replace(raw,price_basis='adjusted_close',bars=tuple(replace(b,adjusted_close=50,split_coefficient=1) for b in raw.bars))
        m,_=calculate_metrics(adjusted,2)
        self.assertEqual(m.period_return_pct,0)
        self.assertEqual(m.max_drawdown_pct,0)

    def test_split_adjusted_return_does_not_treat_raw_volume_as_comparable(self):
        raw=history()
        adjusted=replace(raw,price_basis='adjusted_close',bars=tuple(
            replace(bar,adjusted_close=bar.close,split_coefficient=2 if i==2 else 1)
            for i,bar in enumerate(raw.bars)))
        metrics,warnings=calculate_metrics(adjusted,4)
        self.assertIsNone(metrics.volume_vs_prior_average)
        self.assertTrue(any('split' in warning for warning in warnings))

    def test_bad_calendar_mixed_basis_duplicates_and_synthetic_data_rejected(self):
        bench=history('SPY')
        for candidate in (replace(bench,bars=bench.bars[:2]+bench.bars[3:]), replace(bench,price_basis='adjusted_close')):
            with self.assertRaises(ValueError):calculate_metrics(history(),4,candidate)
        with self.assertRaises(ValueError):calculate_metrics(replace(history(),bars=history().bars+(history().bars[0],)),4)
        with self.assertRaises(ValueError):calculate_metrics(replace(history(),source='dummy_file'),4)


class MarketWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.provider=Mock()
        self.provider.get_history.side_effect=lambda symbol,**kwargs: history(symbol)
        self.collection=Mock()
        self.collection.query.return_value={'documents':[['Volatility measures dispersion of returns.']],
            'metadatas':[[{'source':'Risk.pdf','page':1,'category':'investing'}]],'distances':[[.2]]}
        self.generator=Mock(model='test-model')
        self.generator.generate.return_value=MarketExplanationDraft(summary='The historical return is positive.',findings=[
            MarketFindingDraft(title='Return',observation='The period return is 5%.',interpretation='Historical performance is not a forecast.',
                               metric_fields=['period_return_pct'],reference_ids=[1])],limitations=[])

    def test_api_metrics_one_rag_then_one_llm_and_json_report(self):
        def generate(context,**kwargs):
            self.collection.query.assert_called_once()
            self.assertAlmostEqual(context['metrics']['period_return_pct'],5)
            self.assertEqual(context['retrieved_chunks'][0]['id'],1)
            return MarketExplanationDraft(summary='The historical return is positive.',findings=[
                MarketFindingDraft(title='Return',observation='The period return is 5%.',interpretation='Historical performance is not a forecast.',
                                   metric_fields=['period_return_pct'],reference_ids=[1])],limitations=[])
        self.generator.generate.side_effect=generate
        report=MarketAnalysisAgent(self.provider,collection=self.collection,generator=self.generator).analyze('aapl',sessions=4,benchmark='SPY')
        self.assertEqual(report.status,'complete')
        self.assertIsNone(report.explanation.fallback_reason)
        self.assertEqual(report.explanation.model,'test-model')
        self.assertEqual(self.provider.get_history.call_count,2)
        self.generator.generate.assert_called_once()
        self.assertIn('No news was fetched',report.to_markdown())
        self.assertIn('| Metric | Value |\n|---|---:|\n',report.to_markdown())
        json.dumps(report.to_dict(),allow_nan=False)

    def test_benchmark_failure_preserves_stock_metrics(self):
        self.provider.get_history.side_effect=[history(),MarketDataUnavailable('No benchmark')]
        report=MarketAnalysisAgent(self.provider,use_llm=False).analyze('AAPL',sessions=4,benchmark='SPY')
        self.assertEqual(report.status,'partial')
        self.assertAlmostEqual(report.metrics.period_return_pct,5)
        self.assertIsNone(report.metrics.benchmark_return_pct)

    def test_incompatible_benchmark_is_omitted_not_filled(self):
        self.provider.get_history.side_effect=[history(),replace(history('SPY'),bars=history('SPY').bars[::2])]
        report=MarketAnalysisAgent(self.provider,use_llm=False).analyze('AAPL',sessions=4,benchmark='SPY')
        self.assertIsNone(report.metrics.benchmark_return_pct)
        self.assertEqual(report.status,'partial')

    def test_dummy_quotes_never_become_metrics_and_no_llm_without_data(self):
        self.provider.get_history.side_effect=MarketDataUnavailable('No history')
        self.provider.get_quote.return_value={'symbol':'AAPL','price':'0','is_dummy':True,'source':'dummy_file'}
        report=MarketAnalysisAgent(self.provider,generator=self.generator).analyze('AAPL')
        self.assertEqual(report.status,'unavailable')
        self.assertIsNone(report.metrics)
        self.assertIsNone(report.quote)
        self.generator.generate.assert_not_called()

    def test_quote_only_fallback_is_explicit(self):
        self.provider.get_history.side_effect=MarketDataUnavailable('No history')
        self.provider.get_quote.return_value={'symbol':'AAPL','price':'100','is_dummy':False,'source':'alpha_vantage',
            'latest_trading_day':'2026-10-30','retrieved_at_utc':'2026-10-31T00:00:00Z','freshness':'fresh','cache_age_seconds':0}
        report=MarketAnalysisAgent(self.provider,use_llm=False).analyze('AAPL')
        self.assertEqual(report.status,'partial')
        self.assertIsNone(report.metrics)
        self.assertIn('Dated quote only',report.to_markdown())

    def test_invalid_llm_citation_numeric_claim_or_api_failure_has_labeled_fallback(self):
        for failure in ('reference','number','api'):
            with self.subTest(failure=failure):
                self.generator.reset_mock(side_effect=True)
                draft=MarketExplanationDraft(summary='Historical observation.',findings=[MarketFindingDraft(title='Return',
                    observation='The period return is 999%.' if failure=='number' else 'The period return is 5%.',
                    interpretation='Not a forecast.',metric_fields=['period_return_pct'],reference_ids=[99] if failure=='reference' else [1])],limitations=[])
                self.generator.generate.return_value=draft
                if failure=='api': self.generator.generate.side_effect=RuntimeError('secret')
                report=MarketAnalysisAgent(self.provider,collection=self.collection,generator=self.generator).analyze('AAPL',sessions=4)
                self.assertIsNotNone(report.explanation.fallback_reason)
                self.assertNotIn('secret',report.to_markdown())
                self.assertAlmostEqual(report.metrics.period_return_pct,5)

    def test_requested_adjusted_basis_is_not_silently_replaced_by_raw_history(self):
        self.provider.get_quote.return_value={'is_dummy':True}
        report=MarketAnalysisAgent(self.provider,use_llm=False).analyze('AAPL',adjusted=True)
        self.assertIsNone(report.metrics)
        self.assertEqual(report.price_basis,'adjusted_close')

    def test_invalid_request_stops_before_api(self):
        for kwargs in ({'sessions':0},{'sessions':100},{'benchmark':'AAPL'}):
            with self.assertRaises(ValueError):MarketAnalysisAgent(self.provider,use_llm=False).analyze('AAPL',**kwargs)
        self.provider.get_history.assert_not_called()

    @patch('tools.openai_market.load_dotenv')
    @patch('tools.openai_market.OpenAI')
    def test_market_llm_transport_is_one_strict_request(self,constructor,dotenv):
        constructor.return_value.chat.completions.create.return_value=SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop',message=SimpleNamespace(refusal=None,content=self.generator.generate.return_value.model_dump_json()))])
        generator=OpenAIMarketExplanationGenerator('test-model')
        generator.generate({'metrics':{}},system='Explain metrics.')
        constructor.assert_called_once_with(timeout=45.0,max_retries=0)
        create=constructor.return_value.chat.completions.create
        create.assert_called_once()
        self.assertTrue(create.call_args.kwargs['response_format']['json_schema']['strict'])
        self.assertFalse(create.call_args.kwargs['store'])
