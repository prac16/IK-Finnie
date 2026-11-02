"""Structured OpenAI transport for explaining a completed goals plan."""
import json
import os
from pathlib import Path
from typing import Protocol
from dotenv import load_dotenv
from openai import OpenAI
from tools.llm_runtime import ModelFallbackMixin, LLMRefused
from pydantic import BaseModel, ConfigDict, Field
from tools.llm_privacy import PrivatePayload, redact_text


class GoalFindingDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    goal_id: str
    observation: str
    explanation: str
    next_steps: list[str]
    recommended_scenarios: list[str]
    field_paths: list[str] = Field(min_length=1)
    reference_ids: list[int]


class GoalsExplanationDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: str
    findings: list[GoalFindingDraft] = Field(min_length=1, max_length=20)
    limitations: list[str]


class GoalsExplanationGenerator(Protocol):
    model: str
    def generate(self, context: dict, *, system: str) -> GoalsExplanationDraft: ...


class OpenAIGoalsExplanationGenerator(ModelFallbackMixin):
    def __init__(self, model: str | None = None):
        load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.client = OpenAI(timeout=45.0, max_retries=0)

    def _generate_once(self, context: dict, *, system: str) -> GoalsExplanationDraft:
        payload = PrivatePayload(context, 'goals')
        response = self.client.chat.completions.create(
            model=self.model, messages=[{'role':'system','content':redact_text(system)},
                                       {'role':'user','content':json.dumps(payload.data,allow_nan=False)}],
            response_format={'type':'json_schema','json_schema':{
                'name':'goals_explanation','strict':True,'schema':GoalsExplanationDraft.model_json_schema()}},
            max_completion_tokens=5000, store=False,
        )
        if not response.choices:
            raise ValueError('Empty goals explanation')
        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMRefused('Model declined the request')
        if choice.finish_reason != 'stop' or choice.message.refusal or not choice.message.content:
            raise ValueError('Incomplete or refused goals explanation')
        draft = GoalsExplanationDraft.model_validate_json(choice.message.content)
        return GoalsExplanationDraft.model_validate(payload.restore(draft.model_dump()))
