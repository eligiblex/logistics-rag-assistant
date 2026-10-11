"""Persistent Chroma index with idempotent synchronization and bounded retrieval."""
from threading import RLock
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from .config import config

class VectorStoreManager:
    def __init__(self, settings=None, embeddings=None):
        self.settings = settings or config
        self.lock = RLock()
        self.embeddings = embeddings if embeddings is not None else OllamaEmbeddings(
            model=self.settings.model.embedding,
            base_url=self.settings.ollama.base_url,
            client_kwargs={
                "timeout": self.settings.ollama.timeout,
                "trust_env": self.settings.ollama.trust_env,
            },
        )
        self.vectorstore = None

    def load_vectorstore(self):
        with self.lock:
            if self.vectorstore is None:
                options = self.settings.vector_store
                self.vectorstore = Chroma(
                    collection_name=options.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.settings.resolve(options.persist_directory)),
                    collection_metadata={"hnsw:space": options.similarity_metric,
                                         "embedding_model": self.settings.model.embedding},
                )
                recorded = (self.vectorstore._collection.metadata or {}).get("embedding_model")
                if recorded and recorded != self.settings.model.embedding:
                    raise ValueError("Embedding model changed: use a new CHROMA_DIRECTORY.")
            return self.vectorstore

    def create_vectorstore(self, documents):
        # Add before delete. This is idempotent, but not a cross-batch transaction.
        with self.lock:
            store = self.load_vectorstore()
            desired = {doc.metadata["id"]: doc for doc in documents}
            if not desired:
                raise ValueError("Refusing to replace the knowledge base with empty input")
            existing = set(store.get(include=[])["ids"])
            new_ids = sorted(set(desired) - existing)
            batch_size = self.settings.vector_store.embedding_batch_size
            for offset in range(0, len(new_ids), batch_size):
                batch = new_ids[offset:offset + batch_size]
                store.add_documents([desired[key] for key in batch], ids=batch)
            obsolete = sorted(existing - set(desired))
            if obsolete:
                store.delete(ids=obsolete)
            return store

    def similarity_search(self, question):
        with self.lock:
            store = self.load_vectorstore()
            count = store._collection.count()
            if not count:
                return []
            matches = store.similarity_search_with_relevance_scores(
                question, k=min(self.settings.vector_store.k, count)
            )
            return [(doc, float(score)) for doc, score in matches
                    if score >= self.settings.vector_store.score_threshold]

    def get_collection_info(self):
        with self.lock:
            store = self.load_vectorstore()
            return {"collection_name": self.settings.vector_store.collection_name,
                    "document_count": store._collection.count(),
                    "embedding_model": self.settings.model.embedding}
