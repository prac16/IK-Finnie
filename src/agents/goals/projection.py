"""Monthly Decimal cash-flow projections and required-contribution solver."""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext

from agents.goals.models import Goal

CENT = Decimal('0.01')


def cash(value) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def add_months(start: date, months: int) -> date:
    index = start.year * 12 + start.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(start.day, monthrange(year, month + 1)[1]))


def months_available(start: date, deadline: date) -> int:
    months = (deadline.year-start.year)*12 + deadline.month-start.month
    return months - (add_months(start, months) > deadline)


def target_at(goal: Goal, start: date, deadline: date) -> Decimal:
    if goal.target_basis == 'future_dollars':
        return cash(goal.target_amount)
    years = (deadline-start).days / 365.25
    factor = Decimal(str((1 + goal.annual_inflation_pct/100) ** years))
    adjusted=cash(Decimal(str(goal.target_amount)) * factor)
    if adjusted <= 0:
        raise ValueError('Inflation-adjusted target is below one cent; revise the assumption or target')
    return adjusted


@dataclass(frozen=True)
class MonthBalance:
    date: str
    opening_balance: float
    growth: float
    fees: float
    contribution: float
    closing_balance: float


@dataclass(frozen=True)
class Projection:
    start_date: str
    deadline: str
    contribution_months: int
    future_target: float
    ending_balance: float
    contributions_total: float
    growth_total: float
    fees_total: float
    shortfall: float
    surplus: float
    projected_funding_pct: float
    status: str
    schedule: tuple[MonthBalance, ...]


def _simulate(goal: Goal, start: date, deadline: date, monthly: Decimal,
              annual_return_pct: float, *, schedule: bool = False):
    count = months_available(start, deadline)
    gross_monthly = Decimal(str((1 + annual_return_pct / 100) ** (1/12) - 1))
    fee_monthly = Decimal(str(1 - (1 - goal.annual_fee_pct / 100) ** (1/12)))
    balance, growth_total, fees_total = cash(goal.starting_balance), Decimal(0), Decimal(0)
    rows = []
    with localcontext() as ctx:
        ctx.prec = 50
        for month in range(1, count+1):
            growth = cash(balance * gross_monthly)
            fees = cash((balance + growth) * fee_monthly)
            closing = cash(balance + growth - fees + monthly)
            if schedule:
                rows.append(MonthBalance(add_months(start, month).isoformat(), float(balance), float(growth),
                                         float(fees), float(monthly), float(closing)))
            balance = closing
            growth_total += growth
            fees_total += fees
    return balance, cash(growth_total), cash(fees_total), tuple(rows)


def project(goal: Goal, start: date, *, monthly_contribution: float | None = None,
            deadline: date | None = None, annual_return_pct: float | None = None,
            include_schedule: bool = True) -> Projection:
    end = deadline or goal.target_date
    monthly = cash(goal.monthly_contribution if monthly_contribution is None else monthly_contribution)
    if end < start or months_available(start, end) > 600 or monthly < 0:
        raise ValueError('Invalid projection horizon or contribution')
    rate = goal.annual_return_pct if annual_return_pct is None else annual_return_pct
    if not -100 < rate <= 100:
        raise ValueError('Invalid annual return assumption')
    balance, growth, fees, rows = _simulate(goal, start, end, monthly, rate, schedule=include_schedule)
    target = target_at(goal, start, end)
    shortfall, surplus = max(Decimal(0), target-balance), max(Decimal(0), balance-target)
    return Projection(start.isoformat(), end.isoformat(), months_available(start, end), float(target), float(balance),
                      float(monthly*months_available(start,end)), float(growth), float(fees), float(shortfall),
                      float(surplus), float(balance/target*100) if target else 100.0,
                      'funded_under_assumptions' if shortfall == 0 else 'shortfall_under_assumptions', rows)


def required_monthly(goal: Goal, start: date) -> float | None:
    """Find the smallest cent-denominated monthly deposit meeting the deadline target."""
    target = target_at(goal, start, goal.target_date)
    def funded(cents):
        balance, *_ = _simulate(goal, start, goal.target_date, Decimal(cents)/100, goal.annual_return_pct)
        return balance >= target
    if funded(0):
        return 0.0
    if months_available(start, goal.target_date) == 0:
        return None
    low, high = 0, 100
    while high < 10**14 and not funded(high):
        high = min(high*2, 10**14)
    if not funded(high):
        return None
    while low+1 < high:
        middle = (low+high)//2
        if funded(middle): high = middle
        else: low = middle
    return float(Decimal(high)/100)
