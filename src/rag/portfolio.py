"""Plan and execute the portfolio's single post-metric retrieval."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from rag.retrieval import QueryCollection, RetrievalResult, retrieve

if TYPE_CHECKING:
    from agents.portfolio.analysis import PortfolioReport


@dataclass(frozen=True)
class KnowledgeSection:
    topic: str
    retrieval: RetrievalResult


@dataclass(frozen=True)
class RetrievalPlan:
    topics: tuple[str, ...]
    query: str
    category: str


def plan_retrieval(analysis: PortfolioReport) -> RetrievalPlan:
    """Combine goal and measured findings into one search, after calculations."""
    portfolio = analysis.portfolio
    goal = portfolio.goal.strip()
    goal_text = "" if goal.casefold() == "not supplied" else goal[:300]
    goal_lower = goal_text.lower()
    types = {h.asset_type.lower() for h in portfolio.holdings}
    allocation = analysis.asset_allocation_pct
    topics = ["Allocation and concentration"]
    terms = ["portfolio asset allocation diversification"]
    if analysis.top_holding_pct >= 25 or analysis.top_three_pct >= 60:
        terms.append("large position concentration risk")
    if any(weight >= 25 for sector, weight in analysis.sector_allocation_pct.items()
           if sector not in {"Multi-sector", "Unknown", "Not applicable"}):
        terms.append("sector concentration")
    if analysis.crypto_pct > 0:
        topics.append("Speculative exposure")
        terms.append("speculative crypto exposure risk")
    if "retire" in goal_lower:
        topics.append("Retirement allocation")
        terms.append("retirement allocation glide path")
    elif "income" in goal_lower and (allocation.get("Fixed income", 0) > 0 or "dividend equity etf" in types):
        topics.append("Income allocation")
        terms.append("bond dividend income preservation")
    if "tax" in goal_lower:
        topics.append("Tax considerations")
        terms.append("tax efficient fund placement municipal bond tax equivalent yield")
    if any(term in goal_lower for term in ("inflation", "purchasing power")) and types & {"short tips etf", "gold etf", "reit etf"}:
        topics.append("Purchasing power")
        terms.append("inflation protection TIPS gold real estate")
    if any(term in goal_lower for term in ("stability", "preservation", "downside")) and allocation.get("Fixed income", 0) > 0:
        topics.append("Capital preservation")
        terms.append("bond duration interest rate risk capital preservation")
    largest = max(portfolio.holdings, key=lambda h: h.weight_pct)
    query = (
        f"Explain {'; '.join(terms)} for the stated goal: {goal_text or 'not supplied'}. "
        f"Measured holdings: largest {largest.ticker} {analysis.top_holding_pct:.1f}%; "
        f"top three {analysis.top_three_pct:.1f}%; crypto {analysis.crypto_pct:.1f}%."
    )
    return RetrievalPlan(tuple(topics), query, "taxes" if "tax" in goal_lower else "investing")


def retrieve_portfolio_knowledge(collection: QueryCollection, plan: RetrievalPlan):
    # Exactly one query per report, with one ranked candidate pool and top-k.
    try:
        result = retrieve(collection, plan.query, category=plan.category, top_k=5, candidate_k=20)
        if result.chunks:
            return (KnowledgeSection("Portfolio goal and measured findings", result),), ()
        return (), ("No usable excerpts for the portfolio goal and measured findings.",)
    except Exception:
        return (), ("Retrieval unavailable for the portfolio goal and measured findings; no supporting excerpts attached.",)
