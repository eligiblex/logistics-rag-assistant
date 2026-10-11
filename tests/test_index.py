import hashlib
from pathlib import Path
import pytest
from pypdf import PdfWriter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import SystemConfig
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager


class TestEmbeddings(Embeddings):
    __test__ = False
    batch_sizes: list[int] = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(v + 1) for v in hashlib.sha256(text.encode()).digest()[:8]]


def settings(tmp_path):
    options = SystemConfig()
    options.knowledge_base.data_directory = str(tmp_path / 'data')
    options.vector_store.persist_directory = str(tmp_path / 'index')
    Path(options.knowledge_base.data_directory).mkdir()
    return options


def test_loader_once_relative_source_and_stable_ids(tmp_path):
    options = settings(tmp_path)
    root = Path(options.knowledge_base.data_directory)
    (root / 'one.txt').write_text('运单内容。', encoding='utf-8')
    (root / 'two.md').write_text('# 仓储\n先进先出。', encoding='utf-8')
    (root / 'ignored.csv').write_text('not loaded')
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(root / 'blank.pdf')
    processor = DocumentProcessor(options)
    docs = processor.process_documents()
    assert len(docs) == 2
    assert {d.metadata['source'] for d in docs} == {'one.txt', 'two.md'}
    assert [d.metadata['id'] for d in docs] == [d.metadata['id'] for d in processor.process_documents()]


def test_refresh_idempotent_and_removes_stale_chunks(tmp_path):
    options = settings(tmp_path)
    path = Path(options.knowledge_base.data_directory) / 'one.txt'
    path.write_text('运单包含收货人信息。', encoding='utf-8')
    processor = DocumentProcessor(options)
    manager = VectorStoreManager(options, TestEmbeddings())
    manager.create_vectorstore(processor.process_documents())
    first_ids = manager.vectorstore.get(include=[])['ids']
    manager.create_vectorstore(processor.process_documents())
    assert manager.vectorstore.get(include=[])['ids'] == first_ids
    path.write_text('仓库采用先进先出管理。', encoding='utf-8')
    manager.create_vectorstore(processor.process_documents())
    assert manager.get_collection_info()['document_count'] == 1
    assert manager.vectorstore.get(include=[])['ids'] != first_ids
    assert manager.similarity_search('仓库')[0][0].page_content == '仓库采用先进先出管理。'
    with pytest.raises(ValueError):
        manager.create_vectorstore([])
    assert manager.get_collection_info()['document_count'] == 1


def test_invalid_overlap():
    with pytest.raises(ValueError):
        SystemConfig.model_validate({'text_splitter': {'chunk_size': 100, 'chunk_overlap': 100}})


def test_embedding_batch_size_is_respected(tmp_path):
    options = settings(tmp_path)
    options.vector_store.embedding_batch_size = 2
    embeddings = TestEmbeddings()
    embeddings.batch_sizes = []
    documents = [
        Document(page_content=f'分块 {index}', metadata={'id': str(index)})
        for index in range(5)
    ]
    VectorStoreManager(options, embeddings).create_vectorstore(documents)
    assert embeddings.batch_sizes == [2, 2, 1]
