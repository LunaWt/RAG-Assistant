from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn
import asyncio
import json


import chromadb
import pytest
import pytest_asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import select

import tests.conftest
import app.main as main_module
import app.services.vector_db as vector_db_module
from app.db import Base, ChatSession, Message, Block
from tests.fakes import FakeEmbeddingModel


def test_list_document_sources(monkeypatch: pytest.MonkeyPatch):
    def fake_list_sources():
        return ["Погода в калифорнии", "Книга о Python"]

    monkeypatch.setattr(
        main_module.vector_db,
        "list_sources",
        fake_list_sources,
    )

    client = TestClient(main_module.app)
    res = client.get("/documents/sources")

    assert res.status_code == 200
    assert res.json() == {"sources": ["Погода в калифорнии", "Книга о Python"]}


@pytest_asyncio.fixture
async def async_engine(
    tmp_path: Path,
) -> AsyncIterator[AsyncEngine]:
    test_db_path = tmp_path / "test.db"
    test_db = create_async_engine(f"sqlite+aiosqlite:///{test_db_path}")

    try:
        async with test_db.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield test_db
    finally:
        await test_db.dispose()


@pytest_asyncio.fixture
async def session_factory(
    async_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def async_client(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as db:
            yield db

    main_module.app.dependency_overrides[main_module.get_db] = override_get_db
    monkeypatch.setattr(
        main_module,
        'SessionLocal',
        session_factory
    )

    try:
        transport = ASGITransport(app=main_module.app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client
    finally:
        main_module.app.dependency_overrides.pop(main_module.get_db, None)


@pytest_asyncio.fixture
async def chat_info(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, Any]:
    user_content = [{
                    'type': 'user-query',
                    'content': 'Приветик, расскажи про калифорнию?',
                    }
                ]
    
    assistant_content = [{
                    'type': 'thought',
                    'content': "I'd need to think about California",
                    },
                    {
                    'type': 'tool',
                    'content': json.dumps({
                        'query': 'Запрос про калифорнию',
                        'hits': 'california.com'
                        }, ensure_ascii=False),
                    },
                    {
                    'type': 'answer',
                    'content': "Калифорния солнечный и технологичный город!",
                    }
                ]

    user_block = [
                    Block(
                        type='user-query',
                        content='Приветик, расскажи про калифорнию?',
                        position=0
                    )
                ]

    positions = [2, 1, 0]

    assistant_blocks_unsorted = [
                        Block(
                            type=b['type'], 
                            content=b['content'], 
                            position=positions[i]
                        )
                        for i, b in enumerate(assistant_content[::-1])
                        ]
    
    async with session_factory() as db:
        chat = ChatSession(title='Чатикс')
        chat.messages = [
            Message(
                role="user", 
                blocks=user_block
            ),
            Message(
                role="assistant", 
                blocks=assistant_blocks_unsorted
            )
        ]
        db.add(chat)
        await db.commit()
        return {
            'chat_id': chat.id, 
            'user_content': user_content, 
            'assistant_content': assistant_content,
            }


@pytest.fixture
def storage_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect uploaded-file storage into tmp_path — never touch app/storage."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    monkeypatch.setattr(main_module.settings, "storage_dir", str(documents_dir))
    return documents_dir


@pytest.fixture
def fake_vector_db(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> vector_db_module.VectorDB:
    """Real VectorDB behaviour on an in-memory Chroma + deterministic embedder."""
    ephemeral_client = chromadb.EphemeralClient()
    monkeypatch.setattr(
        vector_db_module.chromadb,
        "PersistentClient",
        lambda path: ephemeral_client,
    )
    database = vector_db_module.VectorDB(
        path="unused",
        collection_name=f"rag_api_{request.node.name}",
        model_name="test_model",
    )
    database._model = FakeEmbeddingModel()
    monkeypatch.setattr(main_module, "vector_db", database)
    return database


@pytest.mark.asyncio
async def test_add_session_success(
    async_client: AsyncClient,
) -> None:
    create_response = await async_client.post("/sessions")

    assert create_response.status_code == 200

    response_json = create_response.json()

    assert isinstance(response_json["id"], int)
    assert response_json["title"] == "New chat"
    assert response_json["created_at"]

    get_response = await async_client.get("/sessions")

    assert get_response.status_code == 200

    get_json = get_response.json()
    created_at = datetime.fromisoformat(response_json["created_at"])

    assert created_at.utcoffset() == timedelta(0)
    assert get_json == [response_json]


@pytest.mark.asyncio
async def test_db_isolation(
    async_client: AsyncClient,
) -> None:
    response = await async_client.get("/sessions")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_session_messages(
    async_client: AsyncClient,
    chat_info: dict[str, Any]
) -> None:
    chat_id = chat_info['chat_id']
    response = await async_client.get(f"/sessions/{chat_id}/messages")
    resp = response.json()
    messages = resp['messages']
    
    assert response.status_code == 200
    assert resp['id'] == chat_id
    assert resp['title'] == 'Чатикс'

    assert len(messages) == 2
    for idx, message in enumerate(messages):
        if idx == 0:
            assert message['role'] == 'user'
            assert message['blocks'] == chat_info['user_content']
        else:
            assert message['role'] == 'assistant'
            assert message['blocks'] == chat_info['assistant_content']  


@pytest.mark.asyncio
async def test_session_message_not_found(
    async_client: AsyncClient,
) -> None:
    resp = await async_client.get(f"/sessions/{0}/messages")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_cascades(
    async_client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    chat_info: dict[str, Any]
) -> None:
    chat_id = chat_info['chat_id']
    
    delete_chat = await async_client.delete(f"/sessions/{chat_id}")

    assert delete_chat.status_code == 200
    assert delete_chat.json() == {"status": "success", "id": chat_id}

    resp = await async_client.get(f"/sessions/{chat_id}/messages")
    async with session_factory() as db:
        messages = (await db.scalars(select(Message))).all()
        blocks = (await db.scalars(select(Block))).all()
        assert messages == []
        assert blocks == []
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_not_found(
    async_client: AsyncClient,
) -> None:
    resp = await async_client.delete("/sessions/999")

    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_rename_session_success(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
) -> None:
    chat_id = chat_info["chat_id"]

    resp = await async_client.put(
        f"/sessions/{chat_id}",
        json={"title": "  Про Калифорнию  "},
    )

    assert resp.status_code == 200
    # surrounding whitespace is stripped by the endpoint, not by the client
    assert resp.json() == {"id": chat_id, "title": "Про Калифорнию"}

    listed = await async_client.get("/sessions")
    assert [s["title"] for s in listed.json()] == ["Про Калифорнию"]


@pytest.mark.asyncio
async def test_rename_session_truncates_long_title(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
) -> None:
    chat_id = chat_info["chat_id"]

    resp = await async_client.put(f"/sessions/{chat_id}", json={"title": "я" * 100})

    assert resp.status_code == 200
    assert resp.json()["title"] == "я" * 60

    reloaded = await async_client.get(f"/sessions/{chat_id}/messages")
    assert reloaded.json()["title"] == "я" * 60


@pytest.mark.asyncio
async def test_rename_session_rejects_blank_title(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
) -> None:
    chat_id = chat_info["chat_id"]

    resp = await async_client.put(f"/sessions/{chat_id}", json={"title": "   "})

    assert resp.status_code == 422
    assert resp.json()["detail"] == "Title cannot be empty"

    unchanged = await async_client.get(f"/sessions/{chat_id}/messages")
    assert unchanged.json()["title"] == "Чатикс"


@pytest.mark.asyncio
async def test_rename_session_rejects_missing_title(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
) -> None:
    resp = await async_client.put(f"/sessions/{chat_info['chat_id']}", json={})

    assert resp.status_code == 422
    # pydantic validation error, not our explicit HTTPException string
    assert isinstance(resp.json()["detail"], list)


@pytest.mark.asyncio
async def test_rename_session_not_found(
    async_client: AsyncClient,
) -> None:
    resp = await async_client.put("/sessions/999", json={"title": "Новое имя"})

    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_documents_filters_and_limits(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
) -> None:
    fake_vector_db.add_document_to_db(["California", "Weather"], "report.pdf")
    fake_vector_db.add_document_to_db(["Neural"], "other.pdf")

    all_docs = await async_client.get("/documents")
    assert all_docs.status_code == 200
    assert set(all_docs.json()["ids"]) == {
        "report.pdf_chunk_0",
        "report.pdf_chunk_1",
        "other.pdf_chunk_0",
    }

    filtered = await async_client.get("/documents", params={"filename": "report.pdf"})
    assert set(filtered.json()["ids"]) == {
        "report.pdf_chunk_0",
        "report.pdf_chunk_1",
    }

    limited = await async_client.get("/documents", params={"limit": 1})
    assert len(limited.json()["ids"]) == 1


@pytest.mark.asyncio
async def test_preview_document_returns_limited_chunks(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
) -> None:
    fake_vector_db.add_document_to_db(["California", "Weather", "Apple"], "report.pdf")
    fake_vector_db.add_document_to_db(["Neural"], "other.pdf")

    resp = await async_client.get(
        "/documents/preview",
        params={"filename": "report.pdf", "limit": 2},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert len(body["chunks"]) == 2
    assert set(body["chunks"]).issubset({"California", "Weather", "Apple"})


@pytest.mark.asyncio
async def test_preview_strips_path_traversal_from_filename(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
) -> None:
    fake_vector_db.add_document_to_db(["California"], "my report.pdf")

    resp = await async_client.get(
        "/documents/preview",
        params={"filename": "../../etc/my+report.pdf"},
    )

    assert resp.status_code == 200
    assert resp.json()["filename"] == "my report.pdf"


@pytest.mark.asyncio
async def test_preview_unknown_document_returns_404(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
) -> None:
    resp = await async_client.get(
        "/documents/preview",
        params={"filename": "missing.pdf"},
    )

    assert resp.status_code == 404
    assert "missing.pdf" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_document_removes_chunks_and_file(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
    storage_dir: Path,
) -> None:
    fake_vector_db.add_document_to_db(["California", "Weather"], "report.pdf")
    fake_vector_db.add_document_to_db(["Neural"], "other.pdf")
    deleted_file = storage_dir / "report.pdf"
    deleted_file.write_bytes(b"raw pdf bytes")
    kept_file = storage_dir / "other.pdf"
    kept_file.write_bytes(b"raw pdf bytes")

    resp = await async_client.delete("/documents", params={"filename": "report.pdf"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == "report.pdf"
    assert body["chunks_removed"] == 2

    assert not deleted_file.exists()
    assert kept_file.exists()
    assert fake_vector_db.list_sources() == ["other.pdf"]


@pytest.mark.asyncio
async def test_delete_unknown_document_keeps_other_files(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
    storage_dir: Path,
) -> None:
    fake_vector_db.add_document_to_db(["Neural"], "other.pdf")
    kept_file = storage_dir / "other.pdf"
    kept_file.write_bytes(b"raw pdf bytes")

    resp = await async_client.delete("/documents", params={"filename": "missing.pdf"})

    assert resp.status_code == 404
    assert kept_file.exists()
    assert fake_vector_db.list_sources() == ["other.pdf"]


@pytest.mark.asyncio
async def test_upload_document_saves_file_and_indexes_chunks(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
    storage_dir: Path,
) -> None:
    resp = await async_client.post(
        "/upload-document",
        files={
            "file": (
                "report.txt",
                "California weather report".encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["filename"] == "report.txt"
    assert body["chunks"] == 1

    saved_file = storage_dir / "report.txt"
    assert saved_file.read_bytes() == "California weather report".encode("utf-8")
    assert fake_vector_db.list_sources() == ["report.txt"]
    # .strip(): smart_chunk_text still emits a leading "\n\n" on the last chunk
    # (chunker.py:33, scheduled for M04) — don't freeze that defect in an API test
    indexed = [chunk.strip() for chunk in fake_vector_db.preview_chunks("report.txt")]
    assert indexed == ["California weather report"]


@pytest.mark.asyncio
async def test_upload_unsupported_type_returns_422_and_cleans_up(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
    storage_dir: Path,
) -> None:
    resp = await async_client.post(
        "/upload-document",
        files={"file": ("payload.exe", b"MZ binary", "application/octet-stream")},
    )

    assert resp.status_code == 422
    assert resp.json()["detail"] == "Unsupported file type"
    assert not (storage_dir / "payload.exe").exists()
    assert fake_vector_db.list_sources() == []


@pytest.mark.asyncio
async def test_upload_document_fail(
    async_client: AsyncClient,
    fake_vector_db: vector_db_module.VectorDB,
    storage_dir: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_extract_text(filepath: str) -> NoReturn:
        raise RuntimeError()

    monkeypatch.setattr(
        main_module,
        "extract_text",
        fake_extract_text,
    )

    resp = await async_client.post(
        url='/upload-document',
        files={
            "file": ("test.md", b"MZ binary", "application/octet-stream")
        },
    )

    assert resp.status_code == 500
    assert resp.json()['detail'] == 'Ошибка сервера'
    assert list(storage_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_user_gets_events_expected_order(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
            {'type': 'thought_delta', 'text': 'Looking'},
            {'type': 'thought_delta', 'text': 'into db'},
            {
                'type': 'tool_hits', 
                'query': 'California check', 
                'hits': [
                    {
                        'title': 'California weather',
                        'href': 'https://weather.com',
                    }
                ]
            },
            {'type': 'text_delta', 'text': 'The weather in california'},
            {'type': 'text_delta', 'text': 'is sunny.'},
            {'type': 'done'}
        ]
    async def fake_agent_loop(query, history):
        for event in events:
            yield event

    monkeypatch.setattr(
        main_module,
        'agent_loop',
        fake_agent_loop
    )

    async with async_client.stream(
        'POST', 
        '/chat', 
        json={
            'query': ' Weather in California?', 
        }
    ) as response:
        lines = [json.loads(line) async for line in response.aiter_lines()]
        
    assert response.status_code == 200
    assert (response.headers['content-type']).startswith('text/x-ndjson')
    assert lines == events


@pytest.mark.asyncio
async def test_chat_persists_assembled_blocks_after_done(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
            {'type': 'thought_delta', 'text': 'Looking'},
            {'type': 'thought_delta', 'text': ' into db'},
            {
                'type': 'tool_hits', 
                'query': 'California check', 
                'hits': [
                    {
                        'title': 'California weather',
                        'href': 'https://weather.com',
                    }
                ]
            },
            {'type': 'text_delta', 'text': 'The weather in california'},
            {'type': 'text_delta', 'text': ' is sunny.'},
            {'type': 'done'}
        ]
    async def fake_agent_loop(query, history):
        for event in events:
            yield event

    monkeypatch.setattr(
        main_module,
        'agent_loop',
        fake_agent_loop
    )
    session_id = chat_info['chat_id']

    async with async_client.stream(
        'POST', 
        '/chat', 
        json={
            'query': 'Weather in California?',
            'session_id': session_id
            }
    ) as response:
        lines = [json.loads(line) async for line in response.aiter_lines()] 

    messages_response = await async_client.get(
            f'/sessions/{session_id}/messages'
        )
    messages = messages_response.json()['messages']
    user_message_blocks = messages[-2]['blocks']
    assistant_message_blocks = messages[-1]['blocks']
    tool_content = json.dumps({"query": "California check", "hits": [{"title": "California weather", "href": "https://weather.com"}]}, ensure_ascii=False)

    assert messages_response.status_code == 200
    assert len(messages) == 4
    assert user_message_blocks == [{
        'type': 'user-query',
        'content': 'Weather in California?'
        }
    ]
    assert assistant_message_blocks == [
        {
            'type': 'thought', 'content': 'Looking into db'
        },
        {
        "type": "tool",
        "content": tool_content,
        },
        {
            'type': 'answer', 'content': 'The weather in california is sunny.'
        },
    ]


@pytest.mark.asyncio
async def test_chat_does_not_save_assistant_without_done(
    async_client: AsyncClient,
    chat_info: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
            {'type': 'thought_delta', 'text': 'Looking'},
            {'type': 'thought_delta', 'text': ' into db'},
            {
                'type': 'tool_hits', 
                'query': 'California check', 
                'hits': [
                    {
                        'title': 'California weather',
                        'href': 'https://weather.com',
                    }
                ]
            },
            {'type': 'text_delta', 'text': 'The weather in california'},
            {'type': 'text_delta', 'text': ' is sunny.'},
        ]
    async def fake_agent_loop(query, history):
        for event in events:
            yield event

    monkeypatch.setattr(
        main_module,
        'agent_loop',
        fake_agent_loop
    )
    session_id = chat_info['chat_id']

    async with async_client.stream(
        'POST', 
        '/chat', 
        json={
            'query': 'Weather in California?',
            'session_id': session_id
            }
    ) as response:
        async for _ in response.aiter_lines():
           pass

    messages_response = await async_client.get(
            f'/sessions/{session_id}/messages'
        )
    messages = messages_response.json()['messages']

    assert messages_response.status_code == 200
    assert len(messages) == 3
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["blocks"] == [
        {
            "type": "user-query",
            "content": "Weather in California?",
        }
    ]


@pytest.mark.asyncio
async def test_new_chat_title_is_generated_in_background(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch
): 
    session_id = (await async_client.post('/sessions')).json()['id']
    release_title = asyncio.Event()
    received_queries = []

    async def fake_generate_title(query):
        received_queries.append(query)
        await release_title.wait()
        return 'Fake title'

    async def fake_agent_loop(query, history):
        for event in [{'type': 'done'}]:
            yield event

    monkeypatch.setattr(
        main_module,
        'agent_loop',
        fake_agent_loop
    )
    monkeypatch.setattr(
        main_module,
        'generate_title',
        fake_generate_title
    )
    async with asyncio.timeout(2):
        async with async_client.stream(
            'POST',
            '/chat',
            json={'query': 'Whatever', 'session_id': session_id}
        ) as response:
            async for _ in response.aiter_lines():
                pass

    get_old_title = await async_client.get('/sessions')
    assert get_old_title.json()[0]['title'] == 'New chat'

    release_title.set()
    async with asyncio.timeout(1):
        await asyncio.gather(*main_module._background_tasks)

    listed = (await async_client.get('/sessions')).json()
    assert listed[0]['title'] == 'Fake title'
    assert received_queries == ['Whatever']