"""Allow-listed tools. They use demo data and never mutate a real logistics system."""
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class LogisticsToolbox:
    def __init__(self, orders_path: str | Path):
        self.orders_path = Path(orders_path)

    def _orders(self) -> dict[str, dict[str, Any]]:
        records = json.loads(self.orders_path.read_text(encoding="utf-8"))
        return {item["tracking_number"].upper(): item for item in records}

    def lookup_order(self, tracking_number: str) -> dict[str, Any]:
        order = self._orders().get(tracking_number.upper())
        if not order:
            return {"found": False, "tracking_number": tracking_number.upper()}
        return {"found": True, **order}

    @staticmethod
    def assess_exception(order: dict[str, Any]) -> dict[str, Any]:
        if not order.get("found"):
            return {"level": "unknown", "reason": "运单不存在，无法评估"}
        delay = int(order.get("delay_hours", 0))
        status = order.get("status", "unknown")
        level = "high" if delay >= 48 or status in {"lost", "damaged"} else "medium" if delay >= 12 else "low"
        suggestions = {
            "high": ["立即转人工复核", "联系承运商确认责任", "向客户同步处理时限"],
            "medium": ["持续跟踪下一物流节点", "主动告知客户预计延迟"],
            "low": ["按正常流程观察", "保留节点更新记录"],
        }
        return {"level": level, "delay_hours": delay, "reason": order.get("exception_reason") or "暂无异常", "suggestions": suggestions[level]}

    @staticmethod
    def create_demo_escalation(run_id: str, tracking_number: str | None, reason: str) -> dict[str, Any]:
        """Create a local demo ticket only; no external system is called."""
        return {
            "ticket_id": "DEMO-" + uuid4().hex[:8].upper(),
            "run_id": run_id,
            "tracking_number": tracking_number,
            "reason": reason,
            "mode": "local_demo",
        }
