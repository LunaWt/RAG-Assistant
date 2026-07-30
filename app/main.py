import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote_plus

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import Block, ChatSession, Message, SessionLocal, init_db
from app.services.vector_db import vector_db
from app.config import settings
from app.services.agent import agent_loop
from app.services.chunker import smart_chunk_text
from app.services.llm_client import generate_response, generate_title
from app.services.parser import extract_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    warmup = asyncio.create_task(asyncio.to_thread(vector_db.warm_up))
    _background_tasks.add(warmup)
    warmup.add_done_callback(_background_tasks.discard)
    yield


app = FastAPI(
    version="1.0", description="Fastapi endpoints for RAG-agent", lifespan=lifespan
)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


async def get_db():
    async with SessionLocal() as db:
        yield db


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    session_id: int | None = None


class SessionUpdate(BaseModel):
    title: str


def normalize_filename(raw: str | None) -> str:
    """Decode multipart names: URL-encoding and '+' as spaces."""
    if not raw:
        raise ValueError("Missing filename")
    return Path(unquote_plus(raw, encoding="utf-8", errors="replace")).name


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    return f"""User: {query},
            retrieved chunks: {"\n\n".join(retrieved_chunks)}"""


@app.get("/documents")
def search_uploaded_documents(filename: str | None = None, limit: int | None = None):
    if filename:
        return vector_db.get_documents(filename=filename)
    elif limit:
        return vector_db.get_documents(limit=limit)
    else:
        return vector_db.get_documents()


@app.get("/documents/sources")
def list_document_sources():
    return {"sources": vector_db.list_sources()}


@app.get("/documents/preview")
def preview_document_chunks(filename: str, limit: int = 2):
    try:
        name = normalize_filename(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    chunks = vector_db.preview_chunks(name, limit=limit)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks for {name}")
    return {"filename": name, "chunks": chunks}


@app.delete("/documents")
def delete_document(filename: str):
    try:
        name = normalize_filename(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    removed = vector_db.delete_document(name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {name}")

    file_path = Path(settings.storage_dir) / name
    file_path.unlink(missing_ok=True)

    return {
        "status": "success",
        "filename": name,
        "chunks_removed": removed,
        "message": f"Удалён {name} ({removed} чанков)",
    }


@app.get("/ask")
def ask_rag_bot(query: str, filename: str | None = None) -> StreamingResponse:

    if filename:
        retrieved_chunks = vector_db.rag_search(query, filename)
    else:
        retrieved_chunks = vector_db.rag_search(query)

    prompt = build_prompt(query, retrieved_chunks)
    return StreamingResponse(generate_response(prompt), media_type="text/plain")


@app.post("/upload-document")
async def upload_document(file: UploadFile = File()):
    try:
        filename = normalize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    save_dir = Path(settings.storage_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = extract_text(file_path)
        chunks = smart_chunk_text(text, settings.chunk_size, settings.overlap)
        vector_db.add_document_to_db(chunks, filename)
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail='Ошибка сервера') from e

    return {
        "status": "success",
        "filename": filename,
        "chunks": len(chunks),
        "message": f"Файл {filename} успешно загружен и проиндексирован ({len(chunks)} чанков)",
    }


@app.post("/sessions")
async def create_chat_session(db: AsyncSession = Depends(get_db)):
    chat = ChatSession()
    db.add(chat)
    await db.commit()
    return {"id": chat.id, "title": chat.title, "created_at": as_utc(chat.created_at)}


@app.get("/sessions")
async def list_chat_sessions(db: AsyncSession = Depends(get_db)):
    sessions = await db.scalars(select(ChatSession).order_by(ChatSession.id.desc()))
    return [
        {"id": s.id, "title": s.title, "created_at": as_utc(s.created_at)}
        for s in sessions
    ]


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: int, db: AsyncSession = Depends(get_db)):
    chat = await db.scalar(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .options(selectinload(ChatSession.messages).selectinload(Message.blocks))
    )
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {
        "id": chat.id,
        "title": chat.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "blocks": [{
                    "type": b.type, 
                    "content": b.content, 
                } 
                for b in m.blocks],
            }
            for m in chat.messages
        ],
    }


@app.put("/sessions/{session_id}")
async def rename_chat_session(
    session_id: int, body: SessionUpdate, db: AsyncSession = Depends(get_db)
):
    chat = await db.get(ChatSession, session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title cannot be empty")
    chat.title = title[:60]
    await db.commit()
    return {"id": chat.id, "title": chat.title}


@app.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: int, db: AsyncSession = Depends(get_db)):
    chat = await db.get(ChatSession, session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    await db.delete(chat)
    await db.commit()
    return {"status": "success", "id": session_id}


def apply_event_to_blocks(blocks: list[dict], event: dict) -> None:
    """Server-side mirror of UI timeline: glue deltas, keep tool hits as JSON."""
    t = event["type"]
    if t == "thought_delta":
        if blocks and blocks[-1]["type"] == "thought":
            blocks[-1]["content"] += event["text"]
        else:
            blocks.append({"type": "thought", "content": event["text"]})
    elif t == "text_delta":
        if blocks and blocks[-1]["type"] == "answer":
            blocks[-1]["content"] += event["text"]
        else:
            blocks.append({"type": "answer", "content": event["text"]})
    elif t == "tool_hits":
        content = json.dumps(
            {"query": event["query"], "hits": event["hits"]}, ensure_ascii=False
        )
        blocks.append({"type": "tool", "content": content})


async def load_history(session_id: int) -> list[dict] | None:
    """Flat history for the agent: user text + assistant answer blocks. None = no such session."""
    async with SessionLocal() as db:
        chat = await db.scalar(
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages).selectinload(Message.blocks))
        )
    if chat is None:
        return None
    history: list[dict] = []
    for m in chat.messages:
        text = "\n".join(
            b.content
            for b in m.blocks
            if b.type in ("user-query", "answer") and b.content
        )
        if text:
            history.append({"role": m.role, "content": text})
    return history


async def set_session_title(session_id: int, query: str) -> None:
    """Name a fresh session via the LLM; fall back to truncation if it fails."""
    try:
        title = (await generate_title(query)).strip()
    except Exception:
        title = ""
    if not title:
        title = " ".join(query.split())[:40] or "New chat"
    title = title[:60]
    async with SessionLocal() as db:
        chat = await db.get(ChatSession, session_id)
        if chat is not None:
            chat.title = title
            await db.commit()


async def save_message(session_id: int, role: str, blocks: list[dict]) -> None:
    async with SessionLocal() as db:
        db.add(
            Message(
                session_id=session_id,
                role=role,
                blocks=[
                    Block(type=b["type"], content=b["content"], position=i)
                    for i, b in enumerate(blocks)
                ],
            )
        )
        await db.commit()


_background_tasks: set[asyncio.Task] = set()


async def run_agent_to_queue(
    query: str,
    history: list[dict],
    session_id: int | None,
    queue: asyncio.Queue,
) -> None:
    """Producer: run the agent independently of the HTTP connection.

    Streams events into the queue AND owns the DB write, so a dropped client
    never loses the answer. `None` on the queue signals end of stream.
    """
    blocks: list[dict] = []
    try:
        async for event in agent_loop(query, history=history):
            t = event["type"]
            if t == "stream_reset":
                blocks.clear()
            elif t == "done":
                if session_id is not None and blocks:
                    await save_message(session_id, "assistant", blocks)
            else:
                apply_event_to_blocks(blocks, event)
            await queue.put(event)
    finally:
        await queue.put(None)


@app.post("/chat")
async def chat_rag_bot(body: ChatRequest) -> StreamingResponse:
    history = body.history
    if body.session_id is not None:
        history = await load_history(body.session_id)
        if history is None:
            raise HTTPException(
                status_code=404, detail=f"Session not found: {body.session_id}"
            )
        await save_message(
            body.session_id, "user", [{"type": "user-query", "content": body.query}]
        )
        if (
            not history
        ):  # was empty before this message → first message, title in background
            title_task = asyncio.create_task(
                set_session_title(body.session_id, body.query)
            )
            _background_tasks.add(title_task)
            title_task.add_done_callback(_background_tasks.discard)

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        run_agent_to_queue(body.query, history, body.session_id, queue)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    async def event_stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="text/x-ndjson")
