"""Structured OpenAI explanation transport; prompts and validation belong to agents."""
import json
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from tools.llm_runtime import ModelFallbackMixin, LLMRefused
from pydantic import BaseModel, ConfigDict, Field
from tools.llm_privacy import PrivatePayload, redact_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DraftFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    observation: str
    goal_relevance: str
    options: list[str]
    metric_fields: list[str] = Field(min_length=1)
    evidence_refs: list[int]


class ExplanationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    findings: list[DraftFinding] = Field(min_length=1, max_length=5)
    limitations: list[str]


class ExplanationGenerator(Protocol):
    model: str
    def generate(self, context: dict, *, system: str) -> ExplanationDraft: ...


class OpenAIExplanationGenerator(ModelFallbackMixin):
    def __init__(self, model: str | None = None):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # One request, bounded timeout, no automatic SDK retries.
        self.client = OpenAI(timeout=45.0, max_retries=0)

    def _generate_once(self, context: dict, *, system: str) -> ExplanationDraft:
        payload = PrivatePayload(context, 'portfolio')
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": redact_text(system)},
                      {"role": "user", "content": json.dumps(payload.data, allow_nan=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "portfolio_explanation", "strict": True,
                "schema": ExplanationDraft.model_json_schema(),
            }},
            max_completion_tokens=3500,
            store=False,
        )
        if not result.choices:
            raise ValueError("Empty explanation response")
        choice = result.choices[0]
        if choice.message.refusal:
            raise LLMRefused('Model declined the request')
        if choice.finish_reason != "stop" or choice.message.refusal or not choice.message.content:
            raise ValueError("Incomplete or refused explanation")
        return ExplanationDraft.model_validate_json(choice.message.content)
