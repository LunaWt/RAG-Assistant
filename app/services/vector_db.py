import logging
import threading
from typing import Callable

import chromadb
from app.config import settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorDB:
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
            with self._model_lock:  # double-checked: only one thread loads
                if self._model is None:
                    self._model = SentenceTransformer(
                        self.model_name, device=settings.embedding_device
                    )
        return self._model

    def warm_up(self) -> None:
        """Force model load + one encode off the request path (run in background)."""
        try:
            self.model.encode(["warm up"], normalize_embeddings=True)
        except Exception:
            logger.warning("Embedder warm-up failed; it will lazy-load on first use", exc_info=True)

    def add_document_to_db(
        self,
        chunks: list[str],
        filename: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        new_ids = set(ids)

        batch = max(1, settings.embedding_batch_size)
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch):
            vectors = self.model.encode(
                chunks[start : start + batch], normalize_embeddings=True
            ).tolist()
            embeddings.extend(vectors)
            if on_progress:
                on_progress(len(embeddings), len(chunks))

        metadatas = [{"source": filename} for _ in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        current_ids = (
            self.collection.get(
                where={"source": filename},
                include=[],
            ).get("ids")
            or []
        )
        stale_ids = [chunk_id for chunk_id in current_ids if chunk_id not in new_ids]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
            logger.info("Removed %d stale chunks of %s", len(stale_ids), filename)

        logger.info("Indexed %d chunks of %s", len(chunks), filename)

    def rag_search(self, query: str, filename: str = None) -> list[str]:
        query_vector = self.model.encode([query], normalize_embeddings=True).tolist()
        search_params = {"query_embeddings": query_vector, "n_results": 3}
        if filename:
            search_params["where"] = {"source": filename}

        results = self.collection.query(**search_params)
        return results["documents"][0]

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

    def get_documents(self, filename: str | None = None, limit: int | None = None):
        if filename:
            return self.collection.get(
                include=["metadatas"], where={"source": filename}
            )
        if limit:
            return self.collection.get(include=["metadatas"], limit=limit)
        return self.collection.get(include=["metadatas"])


vector_db = VectorDB(
    path=str(Path(settings.storage_dir) / "chroma"),
    collection_name="rag_documents",
    model_name=settings.embedding_model,
)
