"""Coordinate deterministic holdings analysis with local RAG evidence. No quote API."""
from dataclasses import asdict, dataclass, replace
from urllib.parse import quote

from agents.portfolio.analysis import Portfolio, PortfolioAnalysisAgent, PortfolioReport
from agents.portfolio.explanation import EndUserExplanation, explain_report
from tools.openai_explanation import ExplanationGenerator
from rag.ingest import DEFAULT_PDF_DIR
from rag.retrieval import QueryCollection
from rag.portfolio import KnowledgeSection, plan_retrieval, retrieve_portfolio_knowledge


@dataclass(frozen=True)
class GroundedPortfolioReport:
    analysis: PortfolioReport
    knowledge: tuple[KnowledgeSection, ...]
    warnings: tuple[str, ...]
    explanation: EndUserExplanation | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        parts = [self.explanation.to_markdown() if self.explanation else "", self.analysis.to_markdown(), "", "## Financial knowledge from the PDF library",
                 "These retrieved excerpts provide context; the numeric findings above come only from holdings. "
                 "Retrieval similarity is not verification of a recommendation. Calculations are local; the explanation source is identified above."]
        if not self.knowledge:
            parts.append("No financial knowledge excerpts retrieved.")
        citation = 0
        for section in self.knowledge:
            parts.extend(["", f"### {section.topic}"])
            for chunk in section.retrieval.chunks:
                citation += 1
                url = f"{quote(str(DEFAULT_PDF_DIR / chunk.source), safe='/')}#page={chunk.page}"
                parts.append(f"[{citation}] [{chunk.source}, p. {chunk.page}]({url})")
                parts.append("> " + chunk.text.replace("\n", " "))
            parts.append(f"Retrieved {section.retrieval.metrics.selected_count} chunks in "
                         f"{section.retrieval.metrics.retrieval_ms:.1f} ms.")
        parts.extend(f"\nKnowledge availability: {warning}" for warning in self.warnings)
        return "\n\n".join(parts)


def build_report(
    portfolio: Portfolio, collection: QueryCollection | None = None, *,
    use_llm: bool = True, explanation_generator: ExplanationGenerator | None = None,
) -> GroundedPortfolioReport:
    analysis = PortfolioAnalysisAgent._analyze_portfolio(portfolio)
    if collection is None:
        report = GroundedPortfolioReport(analysis, (), ("RAG was not enabled; calculated metrics are available.",))
    else:
        plan = plan_retrieval(analysis)
        knowledge, warnings = retrieve_portfolio_knowledge(collection, plan)
        report = GroundedPortfolioReport(analysis, knowledge, warnings)
    # The explanation consumes the completed report and does not change metrics.
    kwargs = {"use_llm": use_llm}
    if explanation_generator is not None:
        kwargs["generator"] = explanation_generator
    return replace(report, explanation=explain_report(report, **kwargs))
