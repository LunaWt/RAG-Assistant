import asyncio
import json
import logging
import threading
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
from app.services.jobs import jobs
from app.services.llm_client import generate_title
from app.services.parser import SUPPORTED_SUFFIXES, extract_text

logger = logging.getLogger(__name__)


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


STAGING_DIR = ".staging"

_name_locks: dict[str, threading.Lock] = {}
_name_locks_guard = threading.Lock()


def name_lock(filename: str) -> threading.Lock:
    """One lock per document name, so two uploads of the same name commit in turn."""
    with _name_locks_guard:
        return _name_locks.setdefault(filename, threading.Lock())


def clear_staging(staged_path: Path) -> None:
    try:
        staged_path.unlink(missing_ok=True)
        staged_path.parent.rmdir()
    except OSError:
        logger.warning("Could not clear staging directory %s", staged_path.parent)


def index_document(staged_path: Path, filename: str, job_id: str) -> None:
    """Index a staged document and report progress through the job store.

    On success, the staged file is moved into document storage. The terminal job status
    is published only after staging cleanup has been attempted.
    """
    result: dict
    try:
        jobs.update(job_id, status="running", stage="parsing")
        text = extract_text(staged_path)
        jobs.update(job_id, stage="chunking")
        chunks = smart_chunk_text(text, settings.chunk_size, settings.overlap)
        jobs.update(job_id, stage="embedding", total_chunks=len(chunks))
        with name_lock(filename):
            vector_db.add_document_to_db(
                chunks,
                filename,
                on_progress=lambda done, total: jobs.update(
                    job_id, done_chunks=done, total_chunks=total
                ),
            )
            staged_path.replace(Path(settings.storage_dir) / filename)
        result = {"status": "done", "stage": "done", "chunks": len(chunks)}
    except ValueError as e:
        result = {"status": "error", "stage": "failed", "error": str(e)}
    except Exception:
        logger.exception("Indexing failed for %s", filename)
        result = {"status": "error", "stage": "failed", "error": "Ошибка сервера"}
    try:
        clear_staging(staged_path)
    finally:
        jobs.update(job_id, **result)


@app.post("/upload-document", status_code=202)
async def upload_document(file: UploadFile = File()):
    try:
        filename = normalize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=422, detail="Unsupported file type")

    save_dir = Path(settings.storage_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    job_id = jobs.create(filename)
    staged_path = save_dir / STAGING_DIR / job_id / filename
    try:
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        await asyncio.to_thread(staged_path.write_bytes, content)
    except OSError as e:
        jobs.update(job_id, status="error", stage="failed", error="Ошибка сервера")
        raise HTTPException(status_code=500, detail="Ошибка сервера") from e

    task = asyncio.create_task(
        asyncio.to_thread(index_document, staged_path, filename, job_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"job_id": job_id, "filename": filename, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = jobs.snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


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
