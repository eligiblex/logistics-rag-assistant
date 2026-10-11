"""Validated configuration; relative paths always resolve from the project root."""
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]

class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class ModelSettings(Section):
    llm: str = "qwen2.5:7b"
    embedding: str = "nomic-embed-text"
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)

class OllamaSettings(Section):
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = Field(default=120, ge=1)
    trust_env: bool = False

class VectorSettings(Section):
    collection_name: str = "logistics_knowledge"
    persist_directory: str = "./chroma_db"
    similarity_metric: str = "cosine"
    embedding_batch_size: int = Field(default=1, ge=1, le=128)
    k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0, ge=-1, le=1)

class TextSettings(Section):
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=150, ge=0)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", "。", "；", " ", ""])

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

class ServerSettings(Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=1)

class KnowledgeSettings(Section):
    data_directory: str = "./data"
    supported_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".pdf"])

class LoggingSettings(Section):
    level: str = "INFO"
    file: str = "./logs/logistics_rag.log"

class SystemConfig(Section):
    model: ModelSettings = Field(default_factory=ModelSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vector_store: VectorSettings = Field(default_factory=VectorSettings)
    text_splitter: TextSettings = Field(default_factory=TextSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    knowledge_base: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

class ConfigManager:
    def __init__(self, config_path=None):
        load_dotenv(ROOT / ".env")
        path = Path(config_path) if config_path else ROOT / "config.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        mappings = {
            "OLLAMA_BASE_URL": ("ollama", "base_url"),
            "OLLAMA_TIMEOUT": ("ollama", "timeout"),
            "OLLAMA_TRUST_ENV": ("ollama", "trust_env"),
            "LLM_MODEL": ("model", "llm"),
            "EMBEDDING_MODEL": ("model", "embedding"),
            "SERVER_HOST": ("server", "host"),
            "SERVER_PORT": ("server", "port"),
            "LOG_LEVEL": ("logging", "level"),
            "CHROMA_DIRECTORY": ("vector_store", "persist_directory"),
            "KNOWLEDGE_DATA_DIR": ("knowledge_base", "data_directory"),
        }
        for name, (section, field) in mappings.items():
            if name in os.environ:
                raw.setdefault(section, {})[field] = os.environ[name]
        self.config = SystemConfig.model_validate(raw)

    def get_config(self):
        return self.config

config = ConfigManager().get_config()
