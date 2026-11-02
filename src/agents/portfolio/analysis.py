"""Rules-based portfolio analysis using the supplied sample workbook."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook

from rag.ingest import PROJECT_ROOT
from agents.portfolio.sectors import SECTORS, SECTOR_LABELS, SECTOR_ELIGIBLE_CLASSES

DEFAULT_SAMPLE_WORKBOOK = (
    PROJECT_ROOT / "src" / "data" / "15_sample_investment_portfolios.xlsx"
)
SINGLE_HOLDING_SCREEN_PCT = 10.0
HIGH_CONCENTRATION_PCT = 25.0
HIGH_TOP_THREE_PCT = 60.0

_ASSET_CLASS_BY_TYPE = {
    "bond etf": "Fixed income",
    "short tips etf": "Fixed income",
    "treasury bills etf": "Cash and cash equivalents",
    "long treasury etf": "Fixed income",
    "intermediate treasury etf": "Fixed income",
    "municipal bond etf": "Fixed income",
    "us equity etf": "US equity",
    "s&p 500 etf": "US equity",
    "nasdaq 100 etf": "US equity",
    "small cap value etf": "US equity",
    "dividend equity etf": "US equity",
    "consumer staples etf": "US equity",
    "utilities etf": "US equity",
    "minimum volatility etf": "US equity",
    "international equity etf": "International equity",
    "developed markets etf": "International equity",
    "emerging markets etf": "International equity",
    "global equity etf": "Global equity",
    "reit etf": "Real estate",
    "gold etf": "Commodities",
    "bitcoin etf": "Crypto assets",
    "ether etf": "Crypto assets",
    "stock": "US equity",
    "cash": "Cash and cash equivalents",
}
_EXACT_INDEX_GROUPS = {
    "S&P 500 index": {"SPY", "VOO"},
    "Nasdaq-100 index": {"QQQ", "QQQM"},
}
_BROAD_MARKET_OVERLAP = {
    "VTI": {"SPY", "VOO"},
    "VT": {"VTI", "VXUS", "VEA", "VWO"},
    "VXUS": {"VEA", "VWO"},
}


@dataclass(frozen=True)
class Holding:
    ticker: str
    asset_type: str
    weight_pct: float
    market_value_usd: float
    asset_class_override: str | None = None
    sector: str | None = None
    sector_source: str | None = None

    @property
    def asset_class(self) -> str:
        if self.asset_class_override:
            if self.asset_class_override not in set(_ASSET_CLASS_BY_TYPE.values()):
                raise ValueError(f"Unknown asset class: {self.asset_class_override}")
            return self.asset_class_override
        asset_class = _ASSET_CLASS_BY_TYPE.get(self.asset_type.strip().lower())
        if asset_class is None:
            raise ValueError(f"Unknown asset type: {self.asset_type}")
        return asset_class

    @property
    def sector_label(self) -> str:
        return self.sector or ("Unknown" if self.asset_class in SECTOR_ELIGIBLE_CLASSES else "Not applicable")


@dataclass(frozen=True)
class Portfolio:
    portfolio_id: int
    name: str
    age: int | None
    risk_tolerance: str
    stated_value_usd: float
    goal: str
    analysis_focus: str
    holdings: tuple[Holding, ...]
    as_of_date: str | None = None
    input_source: str = "Sample workbook (educational data)"


@dataclass(frozen=True)
class PortfolioReport:
    portfolio: Portfolio
    calculated_value_usd: float
    asset_allocation_pct: dict[str, float]
    top_holding_pct: float
    top_three_pct: float
    hhi: float
    effective_holdings: float
    known_technology_stock_pct: float
    crypto_pct: float
    sector_allocation_pct: dict[str, float]
    sector_coverage_pct: float | None
    findings: tuple[str, ...]
    suggestions: tuple[str, ...]
    data_limitations: tuple[str, ...]

    def to_markdown(self) -> str:
        allocation = "\n".join(
            f"- {asset_class}: {weight:.1f}%"
            for asset_class, weight in sorted(self.asset_allocation_pct.items())
        )
        sectors = "\n".join(
            f"- {sector}: {weight:.1f}%"
            for sector, weight in sorted(self.sector_allocation_pct.items())
        )
        coverage = (f"{self.sector_coverage_pct:.1f}%" if self.sector_coverage_pct is not None else "not applicable")
        holdings = sorted(
            self.portfolio.holdings, key=lambda holding: holding.weight_pct, reverse=True
        )
        top_positions = "\n".join(
            f"- {holding.ticker}: {holding.weight_pct:.1f}% "
            f"(${holding.market_value_usd:,.0f})"
            for holding in holdings[:3]
        )
        return "\n".join(
            [
                f"# Portfolio analysis: {self.portfolio.name}",
                "",
                (
                    f"**Profile:** age {self.portfolio.age if self.portfolio.age is not None else 'not supplied'}; "
                    f"{self.portfolio.risk_tolerance} stated risk tolerance; "
                    f"goal: {self.portfolio.goal}."
                ),
                f"**Portfolio value:** ${self.calculated_value_usd:,.2f}",
                "",
                f"**Holdings source:** {self.portfolio.input_source}",
                f"**Valuation date:** {self.portfolio.as_of_date or 'not supplied'}",
                "Metrics calculated from supplied USD market values; no market-price lookup.",
                "",
                "## Calculated allocation",
                allocation,
                "",
                "## Holding-level sector allocation",
                sectors,
                f"Equity/real-estate value assigned a specific sector: {coverage}.",
                "Percentages use total portfolio value. Fund labels describe mandates, not measured constituent exposure.",
                "",
                "## Concentration",
                f"- Largest holding: {self.top_holding_pct:.1f}%",
                f"- Three largest holdings: {self.top_three_pct:.1f}%",
                f"- Concentration index (HHI, 0–1): {self.hhi:.4f}",
                f"- Effective number of holdings (weight-based HHI): "
                f"{self.effective_holdings:.1f} instruments",
                f"- Information Technology individual stocks (supplied labels): "
                f"{self.known_technology_stock_pct:.1f}%",
                f"- Crypto assets: {self.crypto_pct:.1f}%",
                "",
                "### Largest positions",
                top_positions,
                "",
                "## Findings",
                *[f"- {finding}" for finding in self.findings],
                "",
                "## Goal-aware considerations",
                *[f"- {suggestion}" for suggestion in self.suggestions],
                "",
                "## Data limitations",
                *[f"- {limitation}" for limitation in self.data_limitations],
                "",
                (
                    "This is for educational purposes only and should not be "
                    "considered financial advice."
                ),
            ]
        )


class PortfolioAnalysisAgent:
    """Validate sample portfolios, calculate risk screens, and explain findings."""

    def __init__(self, workbook_path: Path = DEFAULT_SAMPLE_WORKBOOK) -> None:
        self.workbook_path = workbook_path
        self._portfolios = self._load_portfolios(workbook_path)

    @property
    def portfolios(self) -> tuple[Portfolio, ...]:
        return self._portfolios

    def analyze(self, portfolio: str | int) -> PortfolioReport:
        if isinstance(portfolio, int):
            matches = [item for item in self._portfolios if item.portfolio_id == portfolio]
        else:
            matches = [
                item
                for item in self._portfolios
                if item.name.casefold() == portfolio.strip().casefold()
            ]
        if not matches:
            raise KeyError(f"No sample portfolio found for {portfolio!r}")
        return self._analyze_portfolio(matches[0])

    def analyze_all(self) -> tuple[PortfolioReport, ...]:
        return tuple(self._analyze_portfolio(item) for item in self._portfolios)

    @staticmethod
    def _load_portfolios(workbook_path: Path) -> tuple[Portfolio, ...]:
        if not workbook_path.is_file():
            raise FileNotFoundError(f"Portfolio workbook not found: {workbook_path}")
        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
        try:
            profile_sheet = workbook["Portfolios"]
            holdings_sheet = workbook["Holdings"]
            profiles = {
                int(row[0]): {
                    "name": str(row[1]),
                    "age": int(row[2]),
                    "risk_tolerance": str(row[3]),
                    "stated_value_usd": float(row[4]),
                    "goal": str(row[5]),
                    "analysis_focus": str(row[6]),
                }
                for row in profile_sheet.iter_rows(min_row=2, values_only=True)
                if row[0] is not None
            }
            holdings_by_id: dict[int, list[Holding]] = defaultdict(list)
            for row in holdings_sheet.iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                portfolio_id = int(row[0])
                if portfolio_id not in profiles:
                    raise ValueError(
                        f"Holding references unknown portfolio ID {portfolio_id}"
                    )
                holdings_by_id[portfolio_id].append(
                    Holding(
                        ticker=str(row[2]).strip().upper(),
                        asset_type=str(row[3]).strip(),
                        weight_pct=float(row[4]),
                        market_value_usd=float(row[5]),
                        sector=str(row[6]).strip() if len(row) > 6 and row[6] else None,
                        sector_source=str(row[7]).strip() if len(row) > 7 and row[7] else None,
                    )
                )
            portfolios = tuple(
                Portfolio(
                    portfolio_id=portfolio_id,
                    holdings=tuple(holdings_by_id.get(portfolio_id, ())),
                    **profile,
                )
                for portfolio_id, profile in profiles.items()
            )
        finally:
            workbook.close()

        for portfolio in portfolios:
            PortfolioAnalysisAgent._validate_portfolio(portfolio)
        return portfolios

    @staticmethod
    def _validate_portfolio(portfolio: Portfolio) -> None:
        if not portfolio.holdings:
            raise ValueError(f"{portfolio.name} has no holdings")
        if not math.isfinite(portfolio.stated_value_usd) or portfolio.stated_value_usd <= 0:
            raise ValueError("Portfolio value must be finite and positive")
        for holding in portfolio.holdings:
            if not re.fullmatch(r"[A-Z0-9.-]{1,15}", holding.ticker):
                raise ValueError(f"Invalid ticker: {holding.ticker!r}")
            if not all(math.isfinite(value) for value in (holding.weight_pct, holding.market_value_usd)):
                raise ValueError("Holding weights and values must be finite")
        if any(
            holding.weight_pct < 0 or holding.market_value_usd < 0
            for holding in portfolio.holdings
        ):
            raise ValueError(f"{portfolio.name} has negative holding data")
        weight_total = sum(holding.weight_pct for holding in portfolio.holdings)
        if abs(weight_total - 100.0) > 0.1:
            raise ValueError(
                f"{portfolio.name} weights total {weight_total:.2f}%, expected 100%"
            )
        market_value_total = sum(
            holding.market_value_usd for holding in portfolio.holdings
        )
        if abs(market_value_total - portfolio.stated_value_usd) > max(
            1.0, portfolio.stated_value_usd * 0.001
        ):
            raise ValueError(
                f"{portfolio.name} holdings total ${market_value_total:,.2f}, "
                f"but profile states ${portfolio.stated_value_usd:,.2f}"
            )
        for holding in portfolio.holdings:
            expected = holding.market_value_usd / market_value_total * 100
            if abs(holding.weight_pct - expected) > 0.1:
                raise ValueError(f"{holding.ticker} supplied weight disagrees with market value")
            holding.asset_class
            if holding.sector_label not in SECTOR_LABELS:
                raise ValueError(f"Unknown sector: {holding.sector_label}")
            eligible = holding.asset_class in SECTOR_ELIGIBLE_CLASSES
            if not eligible and holding.sector_label not in {"Not applicable", "Unknown"}:
                raise ValueError(f"Non-equity holding {holding.ticker} cannot have an equity sector")
            if eligible and holding.sector_label == "Not applicable":
                raise ValueError(f"Equity holding {holding.ticker} must use a sector, Multi-sector, or Unknown")
            if holding.asset_type.casefold() == "stock" and holding.sector_label == "Multi-sector":
                raise ValueError("Individual stocks cannot use Multi-sector")

    @staticmethod
    def _analyze_portfolio(portfolio: Portfolio) -> PortfolioReport:
        PortfolioAnalysisAgent._validate_portfolio(portfolio)
        total = sum(h.market_value_usd for h in portfolio.holdings)
        merged: dict[str, Holding] = {}
        for holding in portfolio.holdings:
            prior = merged.get(holding.ticker)
            if prior and (prior.asset_class != holding.asset_class or prior.asset_type.casefold() != holding.asset_type.casefold() or prior.sector_label != holding.sector_label):
                raise ValueError(f"Conflicting classifications for {holding.ticker}")
            value = holding.market_value_usd + (prior.market_value_usd if prior else 0)
            merged[holding.ticker] = replace(holding, market_value_usd=value, weight_pct=value * 100 / total)
        portfolio = replace(portfolio, holdings=tuple(h for h in merged.values() if h.market_value_usd > 0))
        allocations: dict[str, float] = defaultdict(float)
        for holding in portfolio.holdings:
            allocations[holding.asset_class] += holding.weight_pct
        sorted_weights = sorted(
            (holding.weight_pct for holding in portfolio.holdings), reverse=True
        )
        hhi = sum((weight / 100) ** 2 for weight in sorted_weights)
        top_holding_pct = sorted_weights[0]
        top_three_pct = sum(sorted_weights[:3])
        tech_stock_pct = sum(
            holding.weight_pct
            for holding in portfolio.holdings
            if holding.sector_label == "Information Technology" and holding.asset_type.casefold() == "stock"
        )
        sectors: dict[str, float] = defaultdict(float)
        for holding in portfolio.holdings:
            sectors[holding.sector_label] += holding.weight_pct
        eligible_weight = sum(h.weight_pct for h in portfolio.holdings if h.asset_class in SECTOR_ELIGIBLE_CLASSES)
        labeled_weight = sum(h.weight_pct for h in portfolio.holdings if h.sector_label in SECTORS)
        sector_coverage = labeled_weight / eligible_weight * 100 if eligible_weight else None
        crypto_pct = allocations.get("Crypto assets", 0.0)
        findings = list(
            PortfolioAnalysisAgent._concentration_findings(
                portfolio, top_holding_pct, top_three_pct, tech_stock_pct, crypto_pct
            )
        )
        for sector, weight in sorted(sectors.items()):
            if sector in SECTORS and weight >= 25:
                findings.append(f"Holdings labeled {sector} account for {weight:.1f}% of total value; fund mandates are not constituent-level measurements.")
        findings.extend(PortfolioAnalysisAgent._overlap_findings(portfolio))
        findings.extend(PortfolioAnalysisAgent._goal_findings(portfolio, allocations))
        suggestions = list(
            PortfolioAnalysisAgent._goal_suggestions(
                portfolio, top_holding_pct, tech_stock_pct, crypto_pct
            )
        )
        limitations = [
            "Fund look-through holdings and constituent sector percentages are not supplied; "
            "sector allocations use holding labels, and Multi-sector/Unknown exposure is not distributed across sectors.",
            "No return history or correlations are supplied; historical performance and statistical risk are not calculated.",
        ]
        if not any(h.asset_class in {"US equity", "International equity", "Global equity", "Real estate"}
                   and h.asset_type.lower() != "stock" for h in portfolio.holdings):
            limitations[0] = "Supplied sector labels do not establish how holdings move together; correlations are not measured."
        if not portfolio.as_of_date:
            limitations.append("Valuation date is not supplied; current market freshness cannot be verified.")
        if not portfolio.goal.strip() or portfolio.goal == "Not supplied":
            limitations.append("Goal is not supplied; goal alignment is not assessed.")
        if "tax" in portfolio.goal.lower():
            limitations.append(
                "Tax bracket, state, account-by-account holdings, and bond yields are "
                "missing; tax-equivalent yield and asset-location conclusions cannot "
                "be calculated."
            )
        if "retire" in portfolio.goal.lower() or "income" in portfolio.goal.lower():
            limitations.append(
                "Withdrawal rate, spending needs, other income, and required liquidity "
                "are missing; retirement-income adequacy cannot be assessed."
            )
        if any(holding.ticker in {"TLT", "IEF"} for holding in portfolio.holdings):
            limitations.append(
                "Fund durations and interest-rate scenarios are missing; bond price "
                "sensitivity is identified by holdings, not calculated."
            )
        return PortfolioReport(
            portfolio=portfolio,
            calculated_value_usd=sum(
                holding.market_value_usd for holding in portfolio.holdings
            ),
            asset_allocation_pct=dict(allocations),
            top_holding_pct=top_holding_pct,
            top_three_pct=top_three_pct,
            hhi=hhi,
            effective_holdings=1 / hhi if hhi else 0,
            known_technology_stock_pct=tech_stock_pct,
            crypto_pct=crypto_pct,
            sector_allocation_pct=dict(sectors),
            sector_coverage_pct=sector_coverage,
            findings=tuple(findings),
            suggestions=tuple(suggestions),
            data_limitations=tuple(limitations),
        )

    @staticmethod
    def _concentration_findings(
        portfolio: Portfolio,
        top_holding_pct: float,
        top_three_pct: float,
        tech_stock_pct: float,
        crypto_pct: float,
    ) -> Iterable[str]:
        largest_holding = max(
            portfolio.holdings, key=lambda holding: holding.weight_pct
        )
        if top_holding_pct >= HIGH_CONCENTRATION_PCT:
            if largest_holding.asset_type.lower() == "stock":
                yield (
                    f"High single-company concentration: {largest_holding.ticker} is "
                    f"{top_holding_pct:.1f}% of the portfolio."
                )
            else:
                yield (
                    f"{largest_holding.ticker} is the largest instrument at "
                    f"{top_holding_pct:.1f}%; fund-level weight is not equivalent to "
                    "single-company exposure."
                )
        elif top_holding_pct > SINGLE_HOLDING_SCREEN_PCT:
            yield (
                f"The largest instrument, {largest_holding.ticker}, is "
                f"{top_holding_pct:.1f}%; this exceeds the 10% screening threshold "
                "used for this educational concentration check."
            )
        if top_three_pct >= HIGH_TOP_THREE_PCT:
            largest = sorted(portfolio.holdings, key=lambda h: h.weight_pct, reverse=True)[:3]
            if all(h.asset_type.casefold() == "stock" for h in largest):
                yield f"The three largest companies total {top_three_pct:.1f}% of portfolio value."
            else:
                yield (
                    f"The three largest instruments total {top_three_pct:.1f}%; assess "
                    "fund look-through before interpreting this as issuer concentration."
                )
        if tech_stock_pct > 0:
            yield (
                f"Information Technology individual stocks account for {tech_stock_pct:.1f}%; "
                "based on supplied sector labels; this excludes any technology exposure inside ETFs."
            )
        if crypto_pct >= 10:
            yield f"Crypto-related ETFs account for {crypto_pct:.1f}% of the portfolio."

    @staticmethod
    def _overlap_findings(portfolio: Portfolio) -> Iterable[str]:
        weights = {holding.ticker: holding.weight_pct for holding in portfolio.holdings}
        for index_name, members in _EXACT_INDEX_GROUPS.items():
            present = sorted(members & weights.keys())
            if len(present) > 1:
                exposed_weight = sum(weights[ticker] for ticker in present)
                yield (
                    f"Overlapping index exposure: {', '.join(present)} both track the "
                    f"{index_name}, representing {exposed_weight:.1f}% combined weight."
                )
        for ticker, others in _BROAD_MARKET_OVERLAP.items():
            present_others = sorted(others & weights.keys())
            if ticker in weights and present_others:
                yield (
                    f"{ticker} has broad constituent overlap with "
                    f"{', '.join(present_others)}; exact look-through overlap cannot be "
                    "quantified from the supplied data."
                )

    @staticmethod
    def _goal_findings(
        portfolio: Portfolio, allocations: dict[str, float]
    ) -> Iterable[str]:
        defensive_pct = allocations.get("Fixed income", 0) + allocations.get(
            "Cash and cash equivalents", 0
        )
        goal = portfolio.goal.lower()
        if "income and preservation" in goal:
            equity_pct = sum(
                allocations.get(asset_class, 0)
                for asset_class in (
                    "US equity",
                    "International equity",
                    "Global equity",
                )
            )
            yield (
                f"Fixed income and cash comprise {defensive_pct:.1f}% versus "
                f"{equity_pct:.1f}% "
                "equity exposure; this mix emphasizes preservation, but does not by "
                "itself establish income sufficiency."
            )
        if any(holding.ticker in {"TLT", "IEF"} for holding in portfolio.holdings):
            treasury_weight = sum(
                holding.weight_pct
                for holding in portfolio.holdings
                if holding.ticker in {"TLT", "IEF"}
            )
            yield (
                f"Long/intermediate Treasury ETFs represent {treasury_weight:.1f}%; "
                "their prices can be sensitive to interest-rate changes, while exact "
                "sensitivity is unavailable without duration data."
            )
        if any(
            holding.ticker in {"VXUS", "VEA", "VWO"}
            for holding in portfolio.holdings
        ):
            international_pct = allocations.get("International equity", 0)
            yield (
                f"International equity is {international_pct:.1f}%; review regional "
                "overlap and currency exposure within the international funds."
            )
        if any(term in goal for term in ("inflation", "purchasing power")):
            inflation_linked = sum(
                holding.weight_pct
                for holding in portfolio.holdings
                if holding.asset_type.lower()
                in {"short tips etf", "gold etf", "reit etf"}
            )
            yield (
                f"TIPS, gold, and real-estate holdings total {inflation_linked:.1f}%; "
                "these provide different types of inflation exposure and are not a "
                "guarantee of preserving purchasing power."
            )
        if "retire at" in goal and portfolio.age is not None:
            equity_pct = sum(
                allocations.get(asset_class, 0)
                for asset_class in (
                    "US equity",
                    "International equity",
                    "Global equity",
                )
            )
            match = re.search(r"retire at\s+(\d+)", portfolio.goal.lower())
            retirement_age = int(match.group(1)) if match else 65
            years_to_retirement = max(0, retirement_age - portfolio.age)
            yield (
                f"The supplied allocation is {equity_pct:.1f}% equity and "
                f"{defensive_pct:.1f}% fixed income/cash; assess this against the "
                f"{years_to_retirement}-year stated horizon to age {retirement_age} "
                "and an explicit glide path."
            )
        if "income" in goal and any(h.asset_type.lower() == "dividend equity etf" for h in portfolio.holdings):
            dividend_equity = sum(
                holding.weight_pct
                for holding in portfolio.holdings
                if holding.asset_type.lower() == "dividend equity etf"
            )
            yield (
                f"Dividend equity ETFs are {dividend_equity:.1f}%; dividend emphasis "
                "does not by itself establish total return or income reliability."
            )
        if "tax" in goal:
            muni = sum(
                holding.weight_pct
                for holding in portfolio.holdings
                if holding.asset_type.lower() == "municipal bond etf"
            )
            if muni:
                yield (
                    f"Municipal-bond ETF exposure is {muni:.1f}%; compare its "
                    "tax-equivalent yield with alternatives using the investor's actual "
                    "tax rate and state."
                )
            if any(holding.ticker == "SGOV" for holding in portfolio.holdings):
                yield (
                    "The portfolio includes Treasury-bill exposure; review its account "
                    "location and tax treatment alongside the investor's full account map."
                )

    @staticmethod
    def _goal_suggestions(
        portfolio: Portfolio,
        top_holding_pct: float,
        tech_stock_pct: float,
        crypto_pct: float,
    ) -> Iterable[str]:
        if not portfolio.goal.strip() or portfolio.goal.casefold() == "not supplied":
            return
        if top_holding_pct > SINGLE_HOLDING_SCREEN_PCT:
            excess = sum(
                max(0.0, holding.weight_pct - SINGLE_HOLDING_SCREEN_PCT)
                for holding in portfolio.holdings
                if holding.asset_type.lower() == "stock"
            )
            if excess:
                excess_value = portfolio.stated_value_usd * excess / 100
                yield (
                    f"Illustrative concentration scenario: screen individual stocks "
                    f"against a 10% per-position cap. Reallocating the excess would "
                    f"redirect approximately {excess:.1f}% "
                    f"(${excess_value:,.0f}) from individual stocks; choose any "
                    f"replacement allocation to support the stated goal "
                    f"({portfolio.goal}) only after confirming risk tolerance, time "
                    "horizon, and tax consequences."
                )
        if any(
            len(members & {holding.ticker for holding in portfolio.holdings}) > 1
            for members in _EXACT_INDEX_GROUPS.values()
        ):
            yield (
                "Review whether multiple funds tracking the same index are intentional; "
                "consolidation can simplify the portfolio but does not itself diversify it."
            )
        if "income and preservation" in portfolio.goal.lower():
            yield (
                "Stress-test bond duration, inflation protection, cash reserves, and "
                "planned withdrawals; current holdings data contains no yields or "
                "spending requirement."
            )
        if "tax" in portfolio.goal.lower() and any(h.asset_type.lower() == "municipal bond etf" for h in portfolio.holdings):
            yield (
                "Evaluate municipal-bond suitability using tax-equivalent yield and "
                "review asset location with a complete account map and tax professional."
            )
        if tech_stock_pct >= 40:
            yield (
                "If reducing sector concentration is a priority, consider shifting "
                "some single-company exposure toward broader diversified exposures "
                "consistent with the investor's stated goal and risk tolerance once confirmed."
            )
        if crypto_pct >= 10:
            yield (
                "Confirm the portfolio's loss capacity and maximum intended crypto "
                "allocation before adding to or rebalancing these positions."
            )
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sample portfolios using deterministic allocation and risk screens."
    )
    parser.add_argument(
        "portfolio",
        nargs="?",
        help="Portfolio name or ID; omit to analyze every sample portfolio.",
    )
    parser.add_argument(
        "--workbook", type=Path, default=DEFAULT_SAMPLE_WORKBOOK
    )
    parser.add_argument("--holdings", type=Path, help="Uploaded CSV or XLSX holdings")
    parser.add_argument("--rag", action="store_true", help="Attach locally retrieved financial knowledge")
    parser.add_argument("--output", type=Path, help="Write Markdown report")
    parser.add_argument("--no-llm", action="store_true", help="Use local explanations without sending report context to OpenAI")
    args = parser.parse_args()
    from agents.portfolio.holdings import load_holdings
    from agents.portfolio.report import build_report
    from rag.store import local_collection
    if args.holdings:
        portfolios = load_holdings(args.holdings.read_bytes(), args.holdings.name)
    else:
        agent = PortfolioAnalysisAgent(args.workbook)
        selector: str | int | None = args.portfolio
        if isinstance(selector, str) and selector.isdecimal():
            selector = int(selector)
        portfolios = agent.portfolios
        if selector is not None:
            portfolios = tuple(p for p in portfolios if (
                p.portfolio_id == selector if isinstance(selector, int)
                else p.name.casefold() == selector.strip().casefold()
            ))
            if not portfolios:
                raise KeyError(f"No sample portfolio found for {selector!r}")
    collection = local_collection() if args.rag else None
    text = "\n\n---\n\n".join(build_report(p, collection, use_llm=not args.no_llm).to_markdown() for p in portfolios)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
