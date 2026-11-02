"""At most one post-metric financial-concept retrieval for market analysis."""
from rag.retrieval import QueryCollection, retrieve


def retrieve_market_knowledge(collection: QueryCollection, symbol: str, metrics: dict):
    query = (
        f'Explain historical price return, annualized volatility, peak to trough maximum drawdown '
        f'and benchmark outperformance for {symbol}. Measured return {metrics["period_return_pct"]:.2f}%, '
        f'drawdown {metrics["max_drawdown_pct"]:.2f}%. Historical observations are not price forecasts.'
    )
    return retrieve(collection, query, category='investing', top_k=5, candidate_k=20)
