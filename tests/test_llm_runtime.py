import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from decimal import Decimal

from tools.llm_runtime import ModelFallbackMixin, LLMRefused, generate_checked
from agents.market.explanation import _numbers, _supported_numbers


class Generator(ModelFallbackMixin):
    model = 'primary-test'
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
    def _generate_once(self, value):
        self.calls.append(self.model)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class ModelFallbackTests(unittest.TestCase):
    @patch.dict(os.environ, {'OPENAI_FALLBACK_MODEL': 'secondary-test'})
    def test_provider_failure_retries_different_model_once(self):
        generator = Generator([TimeoutError('secret'), 'valid'])
        self.assertEqual(generator.generate('input'), 'valid')
        self.assertEqual(generator.calls, ['primary-test', 'secondary-test'])
        self.assertEqual(generator.attempt_count, 2)

    @patch.dict(os.environ, {'OPENAI_FALLBACK_MODEL': 'secondary-test'})
    def test_validation_failure_also_uses_second_model(self):
        generator = Generator(['invalid', 'valid'])
        def validate(value):
            if value != 'valid': raise ValueError('invalid output')
        self.assertEqual(generate_checked(generator, 'input', validate=validate), 'valid')
        self.assertEqual(len(generator.calls), 2)

    def test_two_failures_stop(self):
        generator = Generator([TimeoutError(), TimeoutError(), 'must not run'])
        with self.assertRaises(TimeoutError): generator.generate('input')
        self.assertEqual(len(generator.calls), 2)

    def test_refusal_does_not_try_to_bypass_with_another_model(self):
        generator = Generator([LLMRefused(), 'must not run'])
        with self.assertRaises(LLMRefused): generator.generate('input')
        self.assertEqual(len(generator.calls), 1)

    @patch.dict(os.environ, {'OPENAI_FALLBACK_MODEL': 'primary-test'})
    def test_same_configured_model_is_not_retried(self):
        generator = Generator([TimeoutError(), 'valid'])
        generator.generate('input')
        self.assertNotEqual(*generator.calls)

    def test_market_date_and_loss_magnitudes_do_not_false_reject(self):
        allowed = _numbers({'date': '2026-10-28', 'return': -5.6, 'benchmark_difference': -3.45})
        self.assertIn(Decimal(28), allowed)
        self.assertTrue(_supported_numbers('On October 28, 2026, it fell 5.60% and lagged the benchmark by 3.45 percentage points.', allowed))
        self.assertFalse(_supported_numbers('A positive return of 3.45%.', allowed))
        self.assertFalse(_supported_numbers('It fell 999%.', allowed))
