"""Load UTF-8 text/Markdown and text-based PDF files exactly once."""
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from .config import config

class DocumentProcessor:
    def __init__(self, settings=None):
        self.settings = settings or config

    def load_documents(self):
        root = self.settings.resolve(self.settings.knowledge_base.data_directory).resolve()
        if not root.is_dir():
            raise FileNotFoundError("Knowledge directory does not exist")
        documents = []
        allowed = self.settings.knowledge_base.supported_extensions
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            path.resolve().relative_to(root)  # Refuse symlinks escaping the knowledge directory.
            source = path.relative_to(root).as_posix()
            if path.suffix.lower() == ".pdf":
                for page_number, page in enumerate(PdfReader(path).pages, start=1):
                    content = page.extract_text() or ""
                    if content.strip():
                        documents.append(Document(page_content=content, metadata={"source": source, "page": page_number}))
            else:
                content = path.read_text(encoding="utf-8")
                if content.strip():
                    documents.append(Document(page_content=content, metadata={"source": source}))
        return documents

    def split_documents(self, documents):
        options = self.settings.text_splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap,
            separators=options.separators,
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index
            identity = f"{chunk.metadata['source']}|{chunk.metadata.get('page', '')}|{index}|{chunk.page_content}"
            chunk.metadata["id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return chunks

    def process_documents(self):
        return self.split_documents(self.load_documents())
