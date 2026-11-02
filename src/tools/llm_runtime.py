"""At most two model attempts, including output validation, with no extra RAG."""
import os
from tools.diagnostics import log_event


class LLMRefused(ValueError):
    """A refusal is respected rather than retried on another model."""


class ModelFallbackMixin:
    def generate(self, *args, **kwargs):
        return self.generate_validated(*args, **kwargs)

    def generate_validated(self, *args, validate=None, **kwargs):
        primary = getattr(self, 'primary_model', self.model)
        self.primary_model = primary
        fallback = os.getenv('OPENAI_FALLBACK_MODEL', '').strip()
        if not fallback or fallback == primary:
            fallback = 'gpt-4o' if primary.startswith('gpt-4o-mini') else 'gpt-4o-mini'
        self.attempt_count = 0
        for index, model in enumerate((primary, fallback), 1):
            self.model = model
            self.attempt_count = index
            stage = 'generation'
            try:
                draft = self._generate_once(*args, **kwargs)
                stage = 'validation'
                if validate:
                    validate(draft)
                log_event('llm_attempt_succeeded', model=model, attempt=index)
                return draft
            except Exception as exc:
                log_event('llm_attempt_failed', model=model, attempt=index, stage=stage, error_type=type(exc).__name__)
                if isinstance(exc, LLMRefused) or index == 2:
                    raise


def generate_checked(generator, *args, validate=None, **kwargs):
    if callable(getattr(type(generator), 'generate_validated', None)):
        return generator.generate_validated(*args, validate=validate, **kwargs)
    # Custom/injected generators keep their existing protocol. Production SDK
    # generators use ModelFallbackMixin; there is no nested retry loop here.
    draft = generator.generate(*args, **kwargs)
    if validate:
        validate(draft)
    return draft
