"""Validated boundaries for the conversational workflow."""
from __future__ import annotations

from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

AgentName = Literal['chat', 'portfolio', 'market', 'goals']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class TurnRequest(Contract):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    message: str = Field(min_length=1, max_length=16000)
    route: AgentName | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    use_llm: bool = True
    use_rag: bool = True

    @field_validator('message')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Message cannot be blank')
        return value.strip()


class MarketArguments(Contract):
    symbol: str
    sessions: int = Field(default=20, strict=True, ge=2, le=99)
    benchmark: str | None = None
    adjusted: bool = Field(default=False, strict=True)

    @field_validator('symbol', 'benchmark')
    @classmethod
    def ticker(cls, value):
        from tools.market_data import normalize_us_symbol
        return normalize_us_symbol(value) if value is not None else None


class GoalsArguments(Contract):
    plan: dict[str, JsonValue] | None = None
    plan_id: str | None = None
    goal_id: str | None = None
    plan_updates: dict[str, JsonValue] = Field(default_factory=dict)
    goal_updates: dict[str, JsonValue] = Field(default_factory=dict)


class PortfolioArguments(Contract):
    portfolio_id: str | None = None


class ChatArguments(Contract):
    question: str | None = None
    category: str | None = None


class RouteDecision(Contract):
    agent: AgentName | None = None
    action: Literal['analyze', 'explain', 'unsupported', 'clarify'] = 'analyze'
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str = ''
    question: str | None = None
    method: str = 'rules'
    router_calls: int = 0


class AgentResult(Contract):
    agent: AgentName
    request_id: str
    status: Literal['success', 'partial', 'needs_input', 'unavailable', 'error']
    markdown: str
    report_id: str | None = None
    input_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)


class TurnResponse(Contract):
    thread_id: str
    request_id: str
    status: Literal['complete', 'needs_input', 'unavailable', 'error']
    message: str
    agent: AgentName | None = None
    result: AgentResult | None = None
    clarification: dict[str, JsonValue] | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
