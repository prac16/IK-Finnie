"""Validated single-currency goal inputs and explicit funding constraints."""
from __future__ import annotations

from datetime import date
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

Money = float


class Goal(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=160)
    target_amount: Money = Field(gt=0, le=1e12)
    target_date: date
    target_basis: Literal['future_dollars', 'today_dollars']
    starting_balance: Money = Field(ge=0, le=1e12)
    monthly_contribution: Money = Field(ge=0, le=1e12)
    annual_return_pct: float = Field(gt=-100, le=100)
    annual_inflation_pct: float = Field(gt=-100, le=100)
    annual_fee_pct: float = Field(default=0, ge=0, lt=100)
    risk_tolerance: Literal['low', 'moderate', 'high']
    priority: int = Field(default=1, strict=True, ge=1, le=10)
    deadline_flexible: bool = False
    target_flexible: bool = False


class GoalPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    as_of_date: date
    currency: Literal['USD'] = 'USD'
    available_savings: Money = Field(ge=0, le=1e12)
    monthly_budget: Money = Field(ge=0, le=1e12)
    goals: list[Goal] = Field(min_length=1, max_length=20)

    @model_validator(mode='after')
    def check_resources(self):
        from decimal import Decimal
        if len({goal.id for goal in self.goals}) != len(self.goals):
            raise ValueError('Goal IDs must be unique')
        total_savings = sum(Decimal(str(g.starting_balance)) for g in self.goals)
        total_contributions = sum(Decimal(str(g.monthly_contribution)) for g in self.goals)
        if total_savings > Decimal(str(self.available_savings)):
            raise ValueError('Assigned starting balances exceed available_savings; money cannot fund two goals at once')
        if total_contributions > Decimal(str(self.monthly_budget)):
            raise ValueError('Assigned contributions exceed monthly_budget; reduce commitments or explicitly revise the budget')
        for goal in self.goals:
            if not goal.name.strip():
                raise ValueError('Goal name cannot be blank')
            if goal.target_date < self.as_of_date:
                raise ValueError(f'{goal.id}: target date precedes the plan date')
            # An explicit technical bound, not an investment recommendation.
            if goal.target_date > _max_date(self.as_of_date):
                raise ValueError(f'{goal.id}: maximum planning horizon is 50 years')
            for amount in (goal.target_amount, goal.starting_balance, goal.monthly_contribution):
                if Decimal(str(amount)) != Decimal(str(amount)).quantize(Decimal('0.01')):
                    raise ValueError('Money amounts must have at most two decimal places')
        for amount in (self.available_savings, self.monthly_budget):
            if Decimal(str(amount)) != Decimal(str(amount)).quantize(Decimal('0.01')):
                raise ValueError('Money amounts must have at most two decimal places')
        return self


def _max_date(start: date) -> date:
    from calendar import monthrange
    if start.year > 9949:
        raise ValueError('Plan date is outside the supported calendar range')
    return date(start.year + 50, start.month, min(start.day, monthrange(start.year + 50, start.month)[1]))
