"""One structured OpenAI request for a market report explanation."""
import json
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from tools.llm_runtime import ModelFallbackMixin, LLMRefused
from pydantic import BaseModel, ConfigDict, Field
from tools.llm_privacy import PrivatePayload, redact_text


class MarketFindingDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str
    observation: str
    interpretation: str
    metric_fields: list[str] = Field(min_length=1)
    reference_ids: list[int]


class MarketExplanationDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: str
    findings: list[MarketFindingDraft] = Field(min_length=1, max_length=5)
    limitations: list[str]


class MarketExplanationGenerator(Protocol):
    model: str
    def generate(self, context: dict, *, system: str) -> MarketExplanationDraft: ...


class OpenAIMarketExplanationGenerator(ModelFallbackMixin):
    def __init__(self, model: str | None = None):
        load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.client = OpenAI(timeout=45.0, max_retries=0)

    def _generate_once(self, context: dict, *, system: str) -> MarketExplanationDraft:
        payload = PrivatePayload(context, 'market')
        schema = MarketExplanationDraft.model_json_schema()
        finding = schema['$defs']['MarketFindingDraft']['properties']
        if payload.data.get('metrics'):
            finding['metric_fields']['items']['enum'] = list(payload.data['metrics'])
        refs = [chunk['id'] for chunk in payload.data.get('retrieved_chunks', [])]
        if refs:
            finding['reference_ids']['items']['enum'] = refs
        else:
            finding['reference_ids']['maxItems'] = 0
        response = self.client.chat.completions.create(
            model=self.model, messages=[{'role': 'system', 'content': redact_text(system)},
                                        {'role': 'user', 'content': json.dumps(payload.data, allow_nan=False)}],
            response_format={'type': 'json_schema', 'json_schema': {
                'name': 'market_explanation', 'strict': True, 'schema': schema,
            }}, max_completion_tokens=2500, store=False,
        )
        if not response.choices:
            raise ValueError('Empty market explanation')
        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMRefused('Model declined the request')
        if choice.finish_reason != 'stop' or choice.message.refusal or not choice.message.content:
            raise ValueError('Incomplete or refused market explanation')
        return MarketExplanationDraft.model_validate_json(choice.message.content)
