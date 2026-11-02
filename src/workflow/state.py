"""Only serializable facts and artifact references enter checkpoints."""
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: dict
    decision: dict
    active_context: dict
    artifact_registry: list[dict]
    validated: dict
    pending_request: dict | None
    result: dict | None
    final_response: str
    metrics: dict
