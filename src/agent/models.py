"""Typed state shared by the agent workflow, API and persistence layer."""
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


RunStatus = Literal["received", "planning", "executing", "waiting_approval", "completed", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceEvent(BaseModel):
    step: str
    detail: str
    timestamp: str = Field(default_factory=utc_now)


class ToolResult(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True


class AgentRun(BaseModel):
    """Serializable workflow state; every transition is saved to SQLite."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    tracking_number: str | None = None
    intent: str = "general"
    status: RunStatus = "received"
    planned_tools: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def add_trace(self, step: str, detail: str) -> None:
        self.trace.append(TraceEvent(step=step, detail=detail))
        self.updated_at = utc_now()
