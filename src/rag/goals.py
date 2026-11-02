"""One goals-specific knowledge query, after projection and scenario calculations."""
from rag.retrieval import QueryCollection, retrieve


def retrieve_goals_knowledge(collection: QueryCollection, results):
    concepts = ['financial goals savings compound interest monthly contributions inflation risk tolerance time horizon']
    if any(result.projection.shortfall for result in results):
        concepts.append('funding shortfall saving more deadline tradeoffs prioritizing goals')
    if any(result.projection.contribution_months <= 36 for result in results):
        concepts.append('short term goals liquidity capital preservation')
    if any(result.goal['target_basis'] == 'today_dollars' for result in results):
        concepts.append('purchasing power future value inflation')
    # Names are untrusted labels; numeric facts come from the projections.
    return retrieve(collection, '; '.join(concepts), top_k=5, candidate_k=20)
