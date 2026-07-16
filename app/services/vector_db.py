import threading

import chromadb
from app.config import settings
from sentence_transformers import SentenceTransformer
from pathlib import Path


class VectorDB():
    def __init__(self, path, collection_name, model_name):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

    @property
    def model(self):
        """Lazy-load the embedder on first use — keeps import/startup fast."""
        if self._model is None:
            with self._model_lock:            # double-checked: only one thread loads
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def warm_up(self) -> None:
        """Force model load + one encode off the request path (run in background)."""
        try:
            self.model.encode(["warm up"], normalize_embeddings=True)
        except Exception as e:
            print(f"⚠️ embedder warm-up failed (will lazy-load on first use): {e}")

    def add_document_to_db(self,
        chunks: list[str],
        filename: str):

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        embeddings = self.model.encode(chunks, normalize_embeddings=True).tolist()

        metadatas = [{"source": filename} for _ in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"✅ В базу добавлено {len(chunks)} чанков из файла {filename}")


    def rag_search(self, query: str, filename: str = None) -> list[str]:
        query_vector = self.model.encode([query], normalize_embeddings=True).tolist()
        search_params = {
            "query_embeddings": query_vector,
            "n_results": 3
        }
        if filename:
            search_params['where'] = {"source": filename}

        results = self.collection.query(**search_params)
        return results['documents'][0]


    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return sorted({m["source"] for m in metadatas if m and m.get("source")})

    def delete_document(self, filename: str) -> int:
        existing = self.collection.get(
            where={"source": filename},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def preview_chunks(self, filename: str, limit: int = 2) -> list[str]:
        data = self.collection.get(
            where={"source": filename},
            include=["documents"],
            limit=limit,
        )
        return data.get("documents") or []

    def get_documents(
        self,
        filename: str | None = None,
        limit: int | None = None
        ):
        if filename:
            return self.collection.get(
                include=['metadatas'],
                where={'source': filename}
            )
        if limit:
            return self.collection.get(
                include=['metadatas'],
                limit=limit
            )
        return self.collection.get(
                include=['metadatas']
            )


vector_db = VectorDB(
    path=str(Path(settings.storage_dir) / 'chroma'),
    collection_name='rag_documents',
    model_name=settings.embedding_model,
    )
