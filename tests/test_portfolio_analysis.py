import csv
import unittest

from agents.portfolio.analysis import (
    DEFAULT_SAMPLE_WORKBOOK,
    PortfolioAnalysisAgent,
)

DATA_DIR = DEFAULT_SAMPLE_WORKBOOK.parent


class PortfolioAnalysisAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = PortfolioAnalysisAgent(DEFAULT_SAMPLE_WORKBOOK)

    def test_loads_and_analyzes_all_sample_portfolios(self) -> None:
        reports = self.agent.analyze_all()

        self.assertEqual(len(reports), 15)
        self.assertEqual(
            {report.portfolio.name for report in reports},
            {
                "Conservative Retiree",
                "Young Aggressive",
                "Balanced Long-Term",
                "Tech Heavy",
                "Dividend Income",
                "ETF Only",
                "Poorly Diversified",
                "International",
                "Bond Heavy",
                "Crypto Heavy",
                "Inflation Hedge",
                "Recession Defensive",
                "Taxable High Income",
                "Target Date Retirement",
                "Complex Multi Asset",
            },
        )
        for report in reports:
            self.assertAlmostEqual(sum(report.asset_allocation_pct.values()), 100)

    def test_sample_workbook_holdings_match_sample_csv(self) -> None:
        csv_path = DATA_DIR / "sample_portfolio_holdings.csv"
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        csv_holdings = {
            (
                int(row["Portfolio ID"]),
                row["Ticker"],
                row["Asset Type"],
                float(row["Weight %"]),
                float(row["Market Value USD"]),
                row["Sector"],
                row["Sector Source"],
            )
            for row in rows
        }
        workbook_holdings = {
            (
                portfolio.portfolio_id,
                holding.ticker,
                holding.asset_type,
                holding.weight_pct,
                holding.market_value_usd,
                holding.sector,
                holding.sector_source,
            )
            for portfolio in self.agent.portfolios
            for holding in portfolio.holdings
        }

        self.assertEqual(workbook_holdings, csv_holdings)

    def test_tech_heavy_portfolio_flags_technology_concentration(self) -> None:
        report = self.agent.analyze("Tech Heavy")

        self.assertEqual(report.known_technology_stock_pct, 75)
        self.assertEqual(report.top_holding_pct, 30)
        self.assertTrue(any("Technology individual stocks account for 75.0%" in item for item in report.findings))
        self.assertTrue(
            any("High single-company concentration" in item for item in report.findings)
        )

    def test_etf_only_portfolio_detects_overlapping_index_funds(self) -> None:
        report = self.agent.analyze("ETF Only")

        self.assertTrue(
            any(
                "SPY, VOO" in finding
                and "60.0% combined weight" in finding
                for finding in report.findings
            )
        )
        self.assertTrue(
            any("VTI has broad constituent overlap" in finding for finding in report.findings)
        )

    def test_retiree_report_evaluates_preservation_without_claiming_income_adequacy(
        self,
    ) -> None:
        report = self.agent.analyze("Conservative Retiree")

        self.assertEqual(report.asset_allocation_pct["Fixed income"], 65)
        self.assertEqual(report.asset_allocation_pct["Cash and cash equivalents"], 15)
        self.assertTrue(
            any(
                "80.0%" in finding and "income sufficiency" in finding
                for finding in report.findings
            )
        )
        self.assertTrue(
            any("Withdrawal rate" in item for item in report.data_limitations)
        )

    def test_taxable_high_income_portfolio_flags_municipals_and_missing_tax_inputs(
        self,
    ) -> None:
        report = self.agent.analyze("Taxable High Income")

        self.assertTrue(
            any("Municipal-bond ETF exposure is 20.0%" in item for item in report.findings)
        )
        self.assertTrue(
            any("Tax bracket, state" in item for item in report.data_limitations)
        )

    def test_bond_heavy_portfolio_flags_rate_sensitivity_limits(self) -> None:
        report = self.agent.analyze("Bond Heavy")

        self.assertTrue(
            any("Long/intermediate Treasury ETFs represent 70.0%" in item for item in report.findings)
        )
        self.assertTrue(
            any("Fund durations" in item for item in report.data_limitations)
        )

    def test_international_portfolio_reports_geographic_overlap(self) -> None:
        report = self.agent.analyze("International")

        self.assertTrue(
            any("VXUS has broad constituent overlap with VEA, VWO" in item for item in report.findings)
        )
        self.assertTrue(any("currency exposure" in item for item in report.findings))

    def test_crypto_heavy_portfolio_quantifies_crypto_allocation(self) -> None:
        report = self.agent.analyze("Crypto Heavy")

        self.assertEqual(report.crypto_pct, 80)
        self.assertTrue(any("Crypto-related ETFs account for 80.0%" in item for item in report.findings))

    def test_target_date_portfolio_compares_allocation_to_stated_horizon(self) -> None:
        report = self.agent.analyze("Target Date Retirement")

        self.assertTrue(
            any("50.0% equity and 50.0% fixed income/cash" in item for item in report.findings)
        )
        self.assertTrue(any("17-year stated horizon" in item for item in report.findings))

    def test_poor_diversification_produces_goal_aware_concentration_scenario(
        self,
    ) -> None:
        report = self.agent.analyze("Poorly Diversified")

        self.assertEqual(report.top_holding_pct, 55)
        self.assertEqual(report.top_three_pct, 100)
        self.assertTrue(
            any(
                "70.0%" in suggestion and "$280,000" in suggestion
                for suggestion in report.suggestions
            )
        )
        self.assertTrue(
            any(
                "stated goal (Growth)" in suggestion
                for suggestion in report.suggestions
            )
        )

    def test_report_contains_disclaimer_and_limitations(self) -> None:
        report = self.agent.analyze(15)

        markdown = report.to_markdown()

        self.assertIn("This is for educational purposes only", markdown)
        self.assertIn("Fund look-through holdings", markdown)
        self.assertIn("Effective number of holdings", markdown)

    def test_unknown_portfolio_is_an_explicit_error(self) -> None:
        with self.assertRaises(KeyError):
            self.agent.analyze("Unknown")


if __name__ == "__main__":
    unittest.main()
