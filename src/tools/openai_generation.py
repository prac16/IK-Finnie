"""OpenAI text-generation transport used by the chat agent."""
from typing import Protocol
from openai import OpenAI
from tools.llm_runtime import ModelFallbackMixin, LLMRefused
from tools.llm_privacy import redact_text

OPENAI_MODEL = "gpt-4o-mini"

class TextGenerator(Protocol):
    def generate(self, *, system: str, prompt: str) -> str: ...


class OpenAIGenerator(ModelFallbackMixin):
    """Generate answers through the OpenAI API."""

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        self.model = model
        self.client = OpenAI(timeout=45.0, max_retries=0)

    def _generate_once(self, *, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": redact_text(system)},
                {"role": "user", "content": redact_text(prompt)},
            ],
            store=False,
        )
        if response.choices and getattr(response.choices[0].message, 'refusal', None):
            raise LLMRefused('Model declined the request')
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise RuntimeError("OpenAI returned an empty or invalid chat response")
        return content.strip()
