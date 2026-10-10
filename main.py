"""Start the local web API; optional Gradio UI is a separate entry mode."""
import argparse
import logging
from src.config import config

def main():
    parser = argparse.ArgumentParser(description="物流知识库 RAG 问答")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mode", choices=["api", "web", "gradio"], default="api")
    args = parser.parse_args()
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    log_path = config.resolve(config.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=config.logging.level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])
    if args.mode == "gradio":
        from src.chat_interface import launch
        launch()
    else:
        from src.web_api import run_server
        run_server()

if __name__ == "__main__":
    main()
