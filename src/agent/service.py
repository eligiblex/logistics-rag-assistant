"""Explicit, inspectable state machine for logistics exception handling."""
import json
import logging
import re
from typing import Any

from .models import AgentRun, ToolResult
from .store import AgentRunStore
from .tools import LogisticsToolbox


logger = logging.getLogger(__name__)


TRACKING_PATTERN = re.compile(r"\b(?:SF|YT|JD|ZTO|YTO)[A-Z0-9-]{6,24}\b", re.IGNORECASE)
EXCEPTION_WORDS = ("延误", "异常", "丢失", "破损", "没到", "停滞", "投诉", "理赔")
SENSITIVE_WORDS = ("创建工单", "升级工单", "转人工", "发起理赔", "投诉承运商")


class AgentService:
    """Plans, executes allow-listed tools, pauses for approval, then resumes."""

    def __init__(self, pipeline: Any, store: AgentRunStore, toolbox: LogisticsToolbox):
        self.pipeline = pipeline
        self.store = store
        self.toolbox = toolbox

    def start(self, query: str, tracking_number: str | None = None) -> AgentRun:
        number = (tracking_number or self._extract_tracking_number(query))
        run = AgentRun(query=query, tracking_number=number.upper() if number else None)
        run.add_trace("received", "已接收请求并创建可恢复运行记录")
        self.store.save(run)
        try:
            self._plan(run)
            self._execute(run)
            if self._needs_approval(run):
                run.status = "waiting_approval"
                run.pending_action = {
                    "tool": "create_demo_escalation",
                    "reason": "用户请求执行工单/投诉/理赔类动作；外部动作必须先确认",
                }
                run.add_trace("approval_gate", "流程暂停，等待人工批准或拒绝")
            else:
                self._finalize(run)
        except Exception as exc:
            logger.exception("Agent run failed: %s", run.run_id)
            run.status = "failed"
            run.error = type(exc).__name__
            run.add_trace("failed", "工作流执行失败；详细异常仅写入服务日志")
        return self.store.save(run)

    def decide(self, run_id: str, approved: bool) -> AgentRun | None:
        run = self.store.get(run_id)
        if run is None or run.status != "waiting_approval":
            return run
        if approved:
            output = self.toolbox.create_demo_escalation(run.run_id, run.tracking_number, run.query)
            run.tool_results.append(ToolResult(name="create_demo_escalation", arguments={}, output=output))
            run.add_trace("human_approval", "人工已批准，创建本地演示工单")
        else:
            run.add_trace("human_approval", "人工已拒绝，未执行工单动作")
        run.pending_action = None
        self._finalize(run, action_approved=approved)
        return self.store.save(run)

    def get(self, run_id: str) -> AgentRun | None:
        return self.store.get(run_id)

    @staticmethod
    def _extract_tracking_number(query: str) -> str | None:
        match = TRACKING_PATTERN.search(query)
        return match.group(0) if match else None

    def _plan(self, run: AgentRun) -> None:
        run.status = "planning"
        run.intent = "exception" if any(word in run.query for word in EXCEPTION_WORDS) else "knowledge"
        tools = ["search_knowledge"]
        if run.tracking_number:
            tools.append("lookup_order")
        if run.intent == "exception" and run.tracking_number:
            tools.append("assess_exception")
        run.planned_tools = tools
        run.add_trace("planning", "根据意图和运单号生成白名单工具计划：" + ", ".join(tools))
        self.store.save(run)

    def _execute(self, run: AgentRun) -> None:
        run.status = "executing"
        inputs, sources = self.pipeline.prepare(run.query)
        run.sources = sources
        run.tool_results.append(ToolResult(
            name="search_knowledge",
            arguments={"query": run.query},
            output={"context": inputs["context"], "match_count": len(sources)},
        ))
        if "lookup_order" in run.planned_tools:
            order = self.toolbox.lookup_order(run.tracking_number or "")
            run.tool_results.append(ToolResult(
                name="lookup_order", arguments={"tracking_number": run.tracking_number}, output=order,
            ))
            if "assess_exception" in run.planned_tools:
                assessment = self.toolbox.assess_exception(order)
                run.tool_results.append(ToolResult(
                    name="assess_exception", arguments={"tracking_number": run.tracking_number}, output=assessment,
                ))
        run.add_trace("tool_execution", f"已执行 {len(run.tool_results)} 个工具并记录输入输出")
        self.store.save(run)

    @staticmethod
    def _needs_approval(run: AgentRun) -> bool:
        return any(word in run.query for word in SENSITIVE_WORDS)

    def _finalize(self, run: AgentRun, action_approved: bool | None = None) -> None:
        context_parts = []
        for result in run.tool_results:
            context_parts.append(f"工具 {result.name} 返回：{json.dumps(result.output, ensure_ascii=False)}")
        instruction = (
            "请依据工具结果回答用户，先给结论，再说明依据和下一步。"
            "不得声称查询了真实物流系统；DEMO 工单必须明确是本地演示。"
        )
        if action_approved is False:
            instruction += "人工拒绝了动作，只给建议，不得声称已创建工单。"
        run.answer = self.pipeline.answer_with_context(run.query, instruction + "\n" + "\n".join(context_parts))
        run.status = "completed"
        run.add_trace("completed", "已基于工具证据生成最终答复")
