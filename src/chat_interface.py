"""Optional Gradio view; display history is not retrieval conversation memory."""
from .config import config
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .rag_pipeline import RAGPipeline

def launch():
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Install requirements-gradio.txt before using --mode gradio") from None
    manager = VectorStoreManager()
    manager.create_vectorstore(DocumentProcessor().process_documents())
    pipeline = RAGPipeline(manager)
    def answer(message, history):
        try:
            result = pipeline.query(message)
            sources = "\n".join(f"[{s['citation']}] {s['source']}" for s in result["sources"])
            return result["answer"] + "\n\n参考资料：\n" + sources
        except Exception:
            return "回答未完成，请检查 Ollama 服务及日志。"
    gr.ChatInterface(answer, type="messages", title="物流知识库 RAG",
                     description="单轮检索问答；显示历史不会参与后续检索。").launch(
                         server_name=config.server.host, server_port=config.server.port)
