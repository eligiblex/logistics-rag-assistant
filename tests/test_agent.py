from pathlib import Path

from fastapi.testclient import TestClient

from src.agent import AgentRunStore, AgentService, LogisticsToolbox
from src.web_api import create_app


class FakePipeline:
    def prepare(self, question):
        sources = [{"citation": 1, "source": "demo.md", "content": "异常件需要记录并跟踪", "chunk_id": "x"}]
        return {"question": question, "context": "异常件需要记录并跟踪"}, sources

    def answer_with_context(self, question, context):
        return "已根据运单、异常评估和知识库生成处理建议。"


def build_service(tmp_path: Path):
    orders = tmp_path / "orders.json"
    orders.write_text(
        '[{"tracking_number":"YT87654321","status":"delayed","delay_hours":36,'
        '"exception_reason":"天气延误","last_event":"中转场滞留"}]',
        encoding="utf-8",
    )
    store = AgentRunStore(tmp_path / "runs.sqlite3")
    return AgentService(FakePipeline(), store, LogisticsToolbox(orders)), store


def test_knowledge_question_finishes_without_approval(tmp_path):
    service, _ = build_service(tmp_path)
    run = service.start("什么是异常件？")
    assert run.status == "completed"
    assert run.planned_tools == ["search_knowledge"]
    assert run.answer


def test_exception_run_uses_order_and_assessment_tools(tmp_path):
    service, _ = build_service(tmp_path)
    run = service.start("运单 YT87654321 为什么延误？")
    assert run.status == "completed"
    assert [item.name for item in run.tool_results] == ["search_knowledge", "lookup_order", "assess_exception"]
    assert run.tool_results[-1].output["level"] == "medium"


def test_sensitive_action_pauses_and_can_resume_after_restart(tmp_path):
    service, store = build_service(tmp_path)
    run = service.start("请为 YT87654321 创建工单处理延误")
    assert run.status == "waiting_approval"
    assert run.pending_action["tool"] == "create_demo_escalation"

    restarted = AgentService(service.pipeline, store, service.toolbox)
    resumed = restarted.decide(run.run_id, approved=True)
    assert resumed.status == "completed"
    ticket = next(item for item in resumed.tool_results if item.name == "create_demo_escalation")
    assert ticket.output["ticket_id"].startswith("DEMO-")
    assert ticket.output["mode"] == "local_demo"


def test_rejected_action_never_calls_action_tool(tmp_path):
    service, _ = build_service(tmp_path)
    run = service.start("请为 YT87654321 发起理赔")
    resumed = service.decide(run.run_id, approved=False)
    assert resumed.status == "completed"
    assert "create_demo_escalation" not in [item.name for item in resumed.tool_results]


def test_agent_api_lifecycle(tmp_path):
    service, _ = build_service(tmp_path)
    with TestClient(create_app(pipeline=FakePipeline(), agent_service=service)) as client:
        created = client.post("/api/agent/runs", json={"query": "请为 YT87654321 转人工处理延误"})
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "waiting_approval"

        fetched = client.get(f"/api/agent/runs/{body['run_id']}")
        assert fetched.status_code == 200
        decided = client.post(
            f"/api/agent/runs/{body['run_id']}/decision", json={"approved": True}
        )
        assert decided.status_code == 200
        assert decided.json()["status"] == "completed"
        assert client.post(
            f"/api/agent/runs/{body['run_id']}/decision", json={"approved": True}
        ).status_code == 409
