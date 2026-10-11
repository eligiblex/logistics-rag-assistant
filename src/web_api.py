"""Local-first API. One worker owns the embedded Chroma collection."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import threading
import time
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .config import ROOT, config
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .rag_pipeline import RAGPipeline
from .agent import AgentRunStore, AgentService, LogisticsToolbox

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()

class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    tracking_number: str | None = Field(default=None, min_length=8, max_length=32)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Query must not be blank")
        return value.strip()

class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool

def create_app(pipeline=None, settings=None, initialize=True, agent_service=None):
    options = settings or config

    @asynccontextmanager
    async def lifespan(app):
        if app.state.pipeline is None and initialize:
            processor = app.state.processor
            manager = VectorStoreManager(options)
            manager.create_vectorstore(processor.process_documents())
            app.state.pipeline = RAGPipeline(manager, options)
        if app.state.agent_service is None and app.state.pipeline is not None:
            store = AgentRunStore(options.resolve("./logs/agent_runs.sqlite3"))
            tools = LogisticsToolbox(ROOT / "data" / "demo_orders.json")
            app.state.agent_service = AgentService(app.state.pipeline, store, tools)
        yield

    app = FastAPI(title="Logistics RAG Assistant", version="1.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.agent_service = agent_service
    app.state.processor = DocumentProcessor(options)
    app.state.started = time.monotonic()
    app.state.refresh_lock = threading.Lock()
    app.state.refresh = {"status": "idle", "last_success": None}
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    def ready():
        if app.state.pipeline is None:
            raise HTTPException(503, "知识库尚未初始化")
        return app.state.pipeline

    def agent_ready():
        if app.state.agent_service is None:
            raise HTTPException(503, "Agent 尚未初始化")
        return app.state.agent_service

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "templates" / "index.html")

    @app.get("/api/health")
    def health():
        ready()
        return {"status": "ready", "note": "Initialization check; not a live model probe"}

    @app.get("/api/status")
    def status():
        ready()
        return {"model": options.model.llm, "embedding_model": options.model.embedding,
                "uptime_seconds": round(time.monotonic() - app.state.started),
                "refresh": dict(app.state.refresh), "conversation_memory": False}

    @app.post("/api/query")
    def query(request: QueryRequest):
        pipeline = ready()
        if request.stream:
            raise HTTPException(400, "流式请求请使用 /api/query/stream")
        try:
            return {**pipeline.query(request.question), "session_id": request.session_id}
        except Exception:
            logger.exception("Query failed")
            raise HTTPException(502, "模型或知识库暂时不可用，请检查服务日志") from None

    @app.post("/api/query/stream")
    def stream(request: QueryRequest):
        pipeline = ready()
        def events():
            try:
                for event in pipeline.stream_events(request.question):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            except Exception:
                logger.exception("Stream failed")
                yield 'data: {"type":"error","message":"回答未完成，请检查模型服务后重试"}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/agent/runs", status_code=201)
    def start_agent(request: AgentRequest):
        run = agent_ready().start(request.query, request.tracking_number)
        return run.model_dump()

    @app.get("/api/agent/runs/{run_id}")
    def get_agent_run(run_id: str):
        run = agent_ready().get(run_id)
        if run is None:
            raise HTTPException(404, "运行记录不存在")
        return run.model_dump()

    @app.post("/api/agent/runs/{run_id}/decision")
    def decide_agent_run(run_id: str, request: ApprovalRequest):
        service = agent_ready()
        before = service.get(run_id)
        if before is None:
            raise HTTPException(404, "运行记录不存在")
        if before.status != "waiting_approval":
            raise HTTPException(409, "当前运行不在等待审批状态")
        return service.decide(run_id, request.approved).model_dump()

    @app.get("/api/knowledge-base/info")
    def knowledge_info():
        return {**ready().vector_store_manager.get_collection_info(), "refresh": dict(app.state.refresh)}

    def refresh_job():
        previous = app.state.refresh.get("last_success")
        try:
            documents = app.state.processor.process_documents()
            ready().vector_store_manager.create_vectorstore(documents)
            app.state.refresh = {"status": "completed", "last_success": datetime.now(timezone.utc).isoformat()}
        except Exception:
            logger.exception("Knowledge refresh failed")
            app.state.refresh = {"status": "failed", "last_success": previous}
        finally:
            app.state.refresh_lock.release()

    @app.post("/api/knowledge-base/refresh", status_code=202)
    def refresh(background_tasks: BackgroundTasks):
        ready()
        if not app.state.refresh_lock.acquire(blocking=False):
            raise HTTPException(409, "知识库正在刷新")
        app.state.refresh = {**app.state.refresh, "status": "running"}
        background_tasks.add_task(refresh_job)
        return {"status": "accepted"}

    return app

app = create_app()

def run_server():
    import uvicorn
    uvicorn.run(app, host=config.server.host, port=config.server.port, workers=1)
