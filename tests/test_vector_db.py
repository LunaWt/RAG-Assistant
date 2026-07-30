from __future__ import annotations

from typing import Any

import chromadb
import pytest

import app.services.vector_db as vector_db_module
from tests.fakes import FakeEmbeddingModel


@pytest.fixture
def vector_db(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> vector_db_module.VectorDB:
    fake_client = chromadb.EphemeralClient()
    monkeypatch.setattr(
        vector_db_module.chromadb,
        "PersistentClient",
        lambda path: fake_client,
    )
    database = vector_db_module.VectorDB(
        path="unused",
        collection_name=f"rag_test_{request.node.name}",
        model_name="test_model",
    )
    database._model = FakeEmbeddingModel()
    return database


def _records_by_id(result: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    return {
        record_id: {
            "document": document,
            "metadata": metadata,
        }
        for record_id, document, metadata in zip(
            result["ids"], result.get("documents", []), result.get("metadatas", [])
        )
    }


def test_add_document_stores_documents_ids_and_source_metadata(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California", "Weather"], "report.pdf")

    result = vector_db.collection.get(include=["documents", "metadatas"])
    records = _records_by_id(result)

    assert records == {
        "report.pdf_chunk_0": {
            "document": "California",
            "metadata": {"source": "report.pdf"},
        },
        "report.pdf_chunk_1": {
            "document": "Weather",
            "metadata": {"source": "report.pdf"},
        },
    }


def test_reupload_replaces_all_existing_chunks(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(
        chunks=["California", "Girls", "we're", "unforgettable"],
        filename="report.pdf",
    )
    vector_db.add_document_to_db(
        chunks=["California", "Gurls"],
        filename="report.pdf",
    )

    data = vector_db.collection.get(
        where={"source": "report.pdf"},
        include=["documents"],
    )
    assert set(data["documents"]) == {"California", "Gurls"}
    assert set(data["ids"]) == {"report.pdf_chunk_0", "report.pdf_chunk_1"}


def test_reupload_does_not_delete_another_filename(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California", "Weather"], "first.pdf")
    vector_db.add_document_to_db(["Neural", "Apple"], "second.pdf")

    vector_db.add_document_to_db(["California"], "first.pdf")

    result = vector_db.collection.get(include=["documents", "metadatas"])
    records = _records_by_id(result)
    assert set(records) == {
        "first.pdf_chunk_0",
        "second.pdf_chunk_0",
        "second.pdf_chunk_1",
    }


def test_list_sources_is_unique_and_sorted(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["Neural"], "z.pdf")
    vector_db.add_document_to_db(["Apple", "Weather"], "a.pdf")
    vector_db.add_document_to_db(["California"], "z.pdf")

    assert vector_db.list_sources() == ["a.pdf", "z.pdf"]


def test_preview_chunks_filters_filename_and_applies_limit(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California", "Weather", "Apple"], "report.pdf")
    vector_db.add_document_to_db(["Neural"], "other.pdf")

    preview = vector_db.preview_chunks("report.pdf", limit=2)

    assert len(preview) == 2
    assert set(preview).issubset({"California", "Weather", "Apple"})


def test_get_documents_supports_filter_and_limit(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California", "Weather"], "report.pdf")
    vector_db.add_document_to_db(["Neural"], "other.pdf")

    all_documents = vector_db.get_documents()
    report_documents = vector_db.get_documents(filename="report.pdf")
    limited_documents = vector_db.get_documents(limit=2)

    assert len(all_documents["ids"]) == 3
    assert set(report_documents["ids"]) == {
        "report.pdf_chunk_0",
        "report.pdf_chunk_1",
    }
    assert {m["source"] for m in report_documents["metadatas"]} == {"report.pdf"}
    assert len(limited_documents["ids"]) == 2


def test_delete_document_only_deletes_source_and_returns_count(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California", "Weather"], "report.pdf")
    vector_db.add_document_to_db(["Neural"], "other.pdf")

    assert vector_db.delete_document("report.pdf") == 2
    assert vector_db.list_sources() == ["other.pdf"]
    assert vector_db.delete_document("missing.pdf") == 0


def test_rag_search_ranks_matching_documents_and_filters_filename(
    vector_db: vector_db_module.VectorDB,
) -> None:
    vector_db.add_document_to_db(["California facts", "Neural networks"], "science.pdf")
    vector_db.add_document_to_db(["Apple nutrition"], "food.pdf")

    results = vector_db.rag_search("California")
    assert results[0] == "California facts"
    assert set(results) == {"California facts", "Neural networks", "Apple nutrition"}
    assert vector_db.rag_search("California", filename="food.pdf") == [
        "Apple nutrition"
    ]
