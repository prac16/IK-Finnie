import json
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from agents.goals.agent import GoalsPlanningAgent
from agents.goals.models import GoalPlan
from agents.goals.projection import add_months, months_available, project, required_monthly
from tools.openai_goals import GoalsExplanationDraft, GoalFindingDraft, OpenAIGoalsExplanationGenerator

DATA=Path(__file__).resolve().parents[1]/'src/data'


def sample(): return json.loads((DATA/'sample_goals_plan.json').read_text())


class GoalsProjectionTests(unittest.TestCase):
    def setUp(self):
        self.plan=GoalPlan.model_validate(sample())
        self.goal=self.plan.goals[0]

    def test_hand_calculated_house_goal(self):
        report=GoalsPlanningAgent(use_llm=False).plan(self.plan)
        result=report.results[0]
        self.assertEqual(result.projection.ending_balance,47400)
        self.assertEqual(result.projection.shortfall,12600)
        self.assertEqual(result.required_monthly,1250)
        self.assertEqual(result.additional_budget_needed,350)
        self.assertEqual(result.projection.contribution_months,36)
        delay=next(s for s in result.scenarios if s.kind=='later_deadline')
        self.assertEqual(delay.deadline,'2031-01-01')
        self.assertEqual(delay.projected_balance,60000)
        self.assertFalse(result.scenarios[0].feasible_with_budget)
        self.assertEqual(report.funding['unassigned_savings'],0)
        self.assertIn('not probability',report.to_markdown())
        json.dumps(report.to_dict(),allow_nan=False)

    def test_growth_formula_matches_monthly_schedule_and_cash_reconciles(self):
        goal=self.goal.model_copy(update={'target_date':date(2027,11,1),'starting_balance':1000,'monthly_contribution':100,'annual_return_pct':12})
        p=project(goal,self.plan.as_of_date)
        r=1.12**(1/12)-1
        expected=1000*(1+r)**12+100*((1+r)**12-1)/r
        self.assertAlmostEqual(p.ending_balance,expected,delta=.10)
        self.assertAlmostEqual(1000+p.contributions_total+p.growth_total-p.fees_total,p.ending_balance,places=2)
        self.assertEqual(p.schedule[0].contribution,100)

    def test_fees_reduce_balance_and_zero_growth_preserves_fee_drag(self):
        goal=self.goal.model_copy(update={'annual_fee_pct':1})
        p=project(goal,self.plan.as_of_date)
        self.assertLess(p.ending_balance,47400)
        self.assertGreater(p.fees_total,0)
        report=GoalsPlanningAgent(use_llm=False).plan(self.plan.model_copy(update={'goals':[goal]}))
        self.assertEqual(report.results[0].zero_growth_projection.ending_balance,p.ending_balance)

    def test_today_dollar_target_inflates_but_future_target_is_not_inflated_twice(self):
        future=self.goal.model_copy(update={'annual_inflation_pct':3})
        today=future.model_copy(update={'target_basis':'today_dollars'})
        self.assertEqual(project(future,self.plan.as_of_date).future_target,60000)
        actual=project(today,self.plan.as_of_date).future_target
        expected=60000*1.03**((today.target_date-self.plan.as_of_date).days/365.25)
        self.assertAlmostEqual(actual,expected,places=2)
        self.assertGreater(required_monthly(today,self.plan.as_of_date),1250)

    def test_required_contribution_is_smallest_cent_solution(self):
        goal=self.goal.model_copy(update={'annual_return_pct':5.0,'annual_fee_pct':.5})
        amount=required_monthly(goal,self.plan.as_of_date)
        self.assertEqual(project(goal,self.plan.as_of_date,monthly_contribution=amount).shortfall,0)
        self.assertGreater(project(goal,self.plan.as_of_date,monthly_contribution=round(amount-.01,2)).shortfall,0)

    def test_already_funded_and_due_today_have_explicit_solutions(self):
        funded=self.goal.model_copy(update={'target_amount':10000})
        self.assertEqual(required_monthly(funded,self.plan.as_of_date),0)
        due=self.goal.model_copy(update={'target_date':self.plan.as_of_date})
        self.assertIsNone(required_monthly(due,self.plan.as_of_date))
        self.assertEqual(project(due,self.plan.as_of_date).ending_balance,15000)
        self.assertEqual(project(due,self.plan.as_of_date).contribution_months,0)

    def test_month_end_leap_day_and_partial_month_conventions(self):
        start=date(2028,1,31)
        self.assertEqual(add_months(start,1),date(2028,2,29))
        self.assertEqual(add_months(start,2),date(2028,3,31))
        self.assertEqual(months_available(start,date(2028,3,30)),1)
        self.assertEqual(months_available(start,date(2028,2,28)),0)

    def test_negative_return_with_no_contributions_has_no_invented_completion_date(self):
        data=sample()
        data['goals'][0].update(monthly_contribution=0,annual_return_pct=-10)
        report=GoalsPlanningAgent(use_llm=False).plan(data)
        delay=next(s for s in report.results[0].scenarios if s.kind=='later_deadline')
        self.assertIsNone(delay.deadline)
        self.assertFalse(delay.feasible_with_budget)


class GoalsValidationAndScenarioTests(unittest.TestCase):
    def test_duplicate_ids_overspending_double_counted_savings_and_bad_rates(self):
        cases=[]
        for field,value in [('available_savings',100),('monthly_budget',100)]:
            data=sample();data[field]=value;cases.append(data)
        for field,value in [('annual_return_pct',-100),('annual_inflation_pct',float('nan')),
                            ('target_amount',-1),('annual_fee_pct',100),('risk_tolerance','unknown'),
                            ('target_date','2020-01-01'),('monthly_contribution',1.001)]:
            data=sample();data['goals'][0][field]=value;cases.append(data)
        data=sample();data.update(available_savings=30000,monthly_budget=1800);data['goals'].append(deepcopy(data['goals'][0]));cases.append(data)
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValidationError):GoalPlan.model_validate(data)

    def test_assumptions_are_required_and_no_market_or_news_tool_is_called(self):
        data=sample();data['goals'][0].pop('annual_return_pct')
        with self.assertRaises(ValidationError):GoalPlan.model_validate(data)
        with patch('tools.market_data.AlphaVantageMarketData',side_effect=AssertionError('No market dependency')):
            GoalsPlanningAgent(use_llm=False).plan(sample())

    def test_fixed_target_and_deadline_are_not_presented_as_allowed_changes(self):
        data=sample();data['goals'][0].update(deadline_flexible=False,target_flexible=False)
        result=GoalsPlanningAgent(use_llm=False).plan(data).results[0]
        for scenario in result.scenarios:
            if scenario.kind in {'later_deadline','smaller_target'}:
                self.assertFalse(scenario.allowed_by_preferences)
                self.assertFalse(scenario.feasible_with_budget)

    def test_multi_goal_ledger_and_priority_transfer_show_donor_impact(self):
        data=json.loads((DATA/'sample_multiple_goals.json').read_text())
        original=deepcopy(data)
        report=GoalsPlanningAgent(use_llm=False).plan(data)
        self.assertEqual(data,original)
        self.assertEqual(report.funding['committed_monthly'],1000)
        self.assertEqual(report.funding['committed_savings'],15000)
        reserve=next(r for r in report.results if r.goal['id']=='reserve')
        transfer=next(s for s in reserve.scenarios if s.kind=='priority_reallocation')
        self.assertEqual(transfer.monthly_contribution,500)
        self.assertEqual(transfer.shortfall,0)
        self.assertEqual(transfer.affected_goals[0]['goal_id'],'travel')
        self.assertEqual(transfer.affected_goals[0]['shortfall'],5000)
        self.assertTrue(transfer.feasible_with_budget)
        self.assertEqual(reserve.projection.shortfall,2400)
        self.assertGreater(report.funding['fixed_contribution_budget_gap'],0)

    def test_extra_savings_scenario_uses_only_unassigned_cash(self):
        data=sample();data['available_savings']=30000
        report=GoalsPlanningAgent(use_llm=False).plan(data)
        result=report.results[0]
        option=next(s for s in result.scenarios if s.kind=='additional_lump_sum')
        self.assertEqual(option.additional_starting_balance,12600)
        self.assertEqual(option.shortfall,0)
        self.assertTrue(option.feasible_with_budget)
        self.assertEqual(result.projection.ending_balance,47400)
        self.assertLessEqual(option.additional_starting_balance,report.funding['unassigned_savings'])

    def test_risk_tolerance_changes_guidance_not_return_assumptions(self):
        data=sample();data['goals'][0]['target_date']='2036-11-01'
        low=GoalsPlanningAgent(use_llm=False).plan(data).results[0]
        data['goals'][0]['risk_tolerance']='high'
        high=GoalsPlanningAgent(use_llm=False).plan(data).results[0]
        self.assertEqual(low.projection.ending_balance,high.projection.ending_balance)
        self.assertNotEqual(low.risk_guidance,high.risk_guidance)
        self.assertIn('capital-preservation',' '.join(low.risk_guidance))
        self.assertIn('growth exposure',' '.join(high.risk_guidance))


class GoalsLLMAndRAGTests(unittest.TestCase):
    def setUp(self):
        self.collection=Mock()
        self.collection.query.return_value={'documents':[['Compound interest and savings goals.']],
            'metadatas':[[{'source':'Goals.pdf','page':1,'category':'personal_finance'}]],'distances':[[.2]]}
        self.generator=Mock(model='test-model')
        self.generator.generate.return_value=self.draft()

    def draft(self):
        return GoalsExplanationDraft(summary='The house goal has a funding gap under the supplied assumptions.',findings=[
            GoalFindingDraft(goal_id='house',observation='The projected balance is 47,400 and the shortfall is 12,600.',
                explanation='The required contribution is 1,250 monthly, above the 900 budget.',
                next_steps=['Compare the allowed deadline and target scenarios.'],recommended_scenarios=['later_deadline','smaller_target'],field_paths=['results.house.projection.shortfall','results.house.required_monthly'],reference_ids=[1])],limitations=[])

    def test_one_retrieval_and_one_llm_after_calculations_with_no_trajectory_in_prompt(self):
        def generate(context,**kwargs):
            self.collection.query.assert_called_once()
            self.assertEqual(context['results']['house']['projection']['ending_balance'],47400)
            self.assertNotIn('schedule',context['results']['house']['projection'])
            self.assertEqual(context['retrieved_chunks'][0]['id'],1)
            self.assertIn('untrusted',kwargs['system'])
            return self.draft()
        self.generator.generate.side_effect=generate
        report=GoalsPlanningAgent(collection=self.collection,generator=self.generator).plan(sample())
        self.generator.generate.assert_called_once()
        self.assertEqual(report.explanation.model,'test-model')
        self.assertIsNone(report.explanation.fallback_reason)
        self.assertEqual(self.collection.query.call_args.kwargs['n_results'],20)

    def test_invalid_numbers_refs_fields_and_api_failure_preserve_calculated_report(self):
        for failure in ('number','reference','field','goal','api'):
            with self.subTest(failure=failure):
                draft=self.draft()
                if failure=='number':draft.findings[0].observation='The target is 99999999.'
                if failure=='reference':draft.findings[0].reference_ids=[99]
                if failure=='field':draft.findings[0].field_paths=['results.house.fake']
                if failure=='goal':draft.findings[0].goal_id='invented'
                self.generator.generate.side_effect=RuntimeError('private-error') if failure=='api' else None
                self.generator.generate.return_value=draft
                report=GoalsPlanningAgent(collection=self.collection,generator=self.generator).plan(sample())
                self.assertIsNotNone(report.explanation.fallback_reason)
                self.assertEqual(report.results[0].projection.ending_balance,47400)
                self.assertNotIn('private-error',report.to_markdown())

    def test_forbidden_or_unaffordable_scenario_recommendation_falls_back(self):
        bad=self.draft()
        bad.findings[0].recommended_scenarios=['required_contribution']
        self.generator.generate.return_value=bad
        report=GoalsPlanningAgent(collection=self.collection,generator=self.generator).plan(sample())
        self.assertIsNotNone(report.explanation.fallback_reason)

    def test_negative_probability_caveat_does_not_trigger_false_guarantee_check(self):
        good=self.draft()
        good.limitations=['These figures are not guarantees or probabilities of success.']
        self.generator.generate.return_value=good
        report=GoalsPlanningAgent(collection=self.collection,generator=self.generator).plan(sample())
        self.assertIsNone(report.explanation.fallback_reason)

    def test_funding_percentage_is_not_presented_as_success_probability(self):
        bad=self.draft();bad.summary='There is a 79% chance of success.'
        self.generator.generate.return_value=bad
        report=GoalsPlanningAgent(collection=self.collection,generator=self.generator).plan(sample())
        self.assertIsNotNone(report.explanation.fallback_reason)

    def test_recommended_priority_transfer_renders_calculated_donor_harm(self):
        data=json.loads((DATA/'sample_multiple_goals.json').read_text())
        findings=[]
        for goal in data['goals']:
            findings.append(GoalFindingDraft(goal_id=goal['id'],observation='A funding gap exists under the stated assumptions.',
                explanation='Compare the goal-specific calculated scenarios.',next_steps=[],
                recommended_scenarios=['priority_reallocation'] if goal['id']=='reserve' else [],
                field_paths=['results.'+goal['id']+'.projection.shortfall'],reference_ids=[]))
        self.generator.generate.return_value=GoalsExplanationDraft(summary='Review competing goals.',findings=findings,limitations=[])
        report=GoalsPlanningAgent(generator=self.generator).plan(data)
        self.assertIsNone(report.explanation.fallback_reason)
        text=report.explanation.to_markdown()
        self.assertIn('Trade-off for travel',text)
        self.assertIn('5,000.00',text)

    def test_retrieval_failure_keeps_metrics_and_local_mode_skips_llm(self):
        self.collection.query.side_effect=RuntimeError('index failed')
        with patch('tools.openai_goals.OpenAI') as client:
            report=GoalsPlanningAgent(collection=self.collection,use_llm=False).plan(sample())
        client.assert_not_called()
        self.collection.query.assert_called_once()
        self.assertTrue(report.warnings)
        self.assertEqual(report.results[0].projection.shortfall,12600)

    @patch('tools.openai_goals.load_dotenv')
    @patch('tools.openai_goals.OpenAI')
    def test_llm_transport_uses_strict_schema_and_one_request(self,constructor,dotenv):
        constructor.return_value.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop',message=SimpleNamespace(content=self.draft().model_dump_json(),refusal=None))])
        generator=OpenAIGoalsExplanationGenerator('test-model')
        generator.generate({'results':{}},system='Explain the plan.')
        constructor.assert_called_once_with(timeout=45.0,max_retries=0)
        call=constructor.return_value.chat.completions.create
        call.assert_called_once()
        self.assertTrue(call.call_args.kwargs['response_format']['json_schema']['strict'])
        self.assertFalse(call.call_args.kwargs['store'])
