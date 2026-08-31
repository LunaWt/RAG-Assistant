# RAG Assistant

A document-grounded chat assistant: FastAPI backend, Streamlit UI, ChromaDB retrieval over
uploaded files, and an agent loop that decides for itself when to search the knowledge base,
search the web, or just answer. The agent loop is written by hand — no LangChain, no
LlamaIndex — so every tool call, retry and stream event is code in this repo.

Writing it by hand is the best way I know to understand what is under the hood of the
frameworks we use every day. It is my default on every project, and it usually makes things
less buggy, because you write and test every piece yourself. Sometimes it makes them worse —
skill issue, I guess. I also never realised before this that 90% of software engineering is
testing and debugging. Hell, it is.

## How it works

```
Streamlit (ui/app.py)
    │  POST /chat  { query, history, session_id }
    │  NDJSON: thought_delta | text_delta | tool_start | tool_hits | stream_reset | done
    ▼
FastAPI (app/main.py)
    │  event_stream → agent_loop(query, history)
    │  background asyncio.Task + Queue: the answer finishes even if the client disconnects
    │  assembled thought/tool/answer blocks are persisted to SQLite after `done`
    ▼
agent_loop (app/services/agent.py)
    │  system prompt + the list of indexed documents
    │  chats.create(history=…) — native model turns, not text injection
    │  send_message_stream(query) ↔ Gemini, tool calls executed manually
    ▼
Tools: vector_db (Chroma) | web_search (ddgs + trafilatura + summarisation) | calculator
```

- **Retrieval:** ChromaDB `PersistentClient`, `BAAI/bge-m3` embeddings (1024d, normalized,
  so L2 ranking is equivalent to cosine). Chunking at 1000 characters with 50 overlap.
- **Parsing:** PDF, TXT/MD, DOCX, XLSX, PPTX; UTF-8 with a cp1251 fallback.
- **Persistence:** async SQLAlchemy over SQLite, `sessions → messages → blocks` with
  cascade deletes. A reloaded conversation renders from the same block structure the
  stream produced.
- **Streaming contract:** NDJSON events, one per line. `stream_reset` clears partial
  blocks on a mid-stream failure; nothing is written to the database before `done`.

## Decisions and trade-offs

- **Own ReAct loop instead of a framework.** A framework would hide exactly the parts that
  break in production: retry classification, the iteration limit, what happens to a partial
  answer when the model 503s mid-stream.
- **Retries are classified, not blanket.** Only `{408, 429, 500, 502, 503, 504}` plus
  `httpx.TransportError`/`TimeoutError` are retried, with backoff
  `min(1.0 · 2ⁿ, 8.0)`. A programming error returns immediately instead of being retried
  ten times behind a spinner.
- **The answer survives a closed tab.** The stream runs as a background task feeding a
  queue, so disconnecting the client does not cancel generation or lose the message.
- **Errors never leak internals to the UI.** `str(e)` goes to the diagnostic log, the user
  gets a generic message — the upload path also unlinks the file when indexing fails.

## Run

```sh
pip install -r requirements.txt
cp .env.example .env          # fill GEMINI_API_KEY and HF_TOKEN

uvicorn app.main:app --reload # API on :8000
cd ui && streamlit run app.py # UI on :8501
```

The first request downloads `bge-m3` (~2 GB) from Hugging Face.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q           # 97 passed
```

No test touches the network, the real Chroma store or a real model: `tests/fakes.py` holds
a scripted chat client and a fake embedder, and `tests/conftest.py` redirects `STORAGE_DIR`
to a temporary directory **at import time** — a fixture-level monkeypatch is one phase too
late, because `vector_db` is constructed at import.

Two bugs were found this way and fixed test-first: an exact-boundary chunk was dropped by
the chunker, and the tool loop ran 21 iterations against a limit of 20 (`<=` instead of `<`).

## Status

Portfolio v1, in progress: 3 of 14 planned milestones are closed (test foundation,
API/persistence coverage, LLM client + agent hardening). Not built yet: authentication
and per-user isolation, cross-encoder reranking, hybrid BM25 + vector retrieval, RAGAS
evaluation, a sandboxed code-execution tool, Celery/Redis background indexing, and the
Docker Compose deployment. Uploaded documents currently have no ownership boundary — this
runs locally, single user, on purpose.
