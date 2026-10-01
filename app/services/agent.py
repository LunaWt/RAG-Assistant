import asyncio
import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from openai import APIConnectionError, APIStatusError

from app.config import settings
from app.services.llm_client import client
from app.services.tools import ToolError, calculator, search_knowledge_base, web_search
from app.services.vision import image_part, render_page
from app.services.vector_db import vector_db

logger = logging.getLogger(__name__)

TOOLS = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over the documents the user uploaded. "
                "Pass filename to search inside a single document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in the user's language.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "One of the indexed document names.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web and return a summary of the pages found, with links."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression exactly, e.g. '2 + 1 * 9 / 16'. "
                "Numeric literals and + - * / ** % only: no variables, no functions. "
                "Up to 100 characters; the result is rounded to 4 decimals."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

RETRYABLE_API_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
RETRIES = 5
BASE_DELAY = 1.0   # seconds before the first retry
MAX_DELAY = 8.0
MAX_ITERATIONS = 20  # tool rounds per answer, before the model is asked to conclude


@dataclass
class ToolCall:
    """One tool call rebuilt from the stream; raw_args is JSON text, not a dict."""

    id: str = ""
    name: str = ""
    raw_args: str = ""
    # Google's thought_signature lives here and goes back exactly as received (1 Oct 2026:
    # Gemma answered the next request with 500 in 2 of 4 tries without it, 0 of 4 with it).
    extra_content: dict | None = None

    @property
    def args(self) -> dict | None:
        """Parsed arguments, or None when the model emitted something that is not a JSON object."""
        try:
            parsed = json.loads(self.raw_args or "{}")
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


@dataclass
class Turn:
    """One model turn: the answer text, the content exactly as streamed, and the tool calls.

    `raw` keeps thoughts that arrive inside the content (Gemma's <thought>...</thought>), so
    the model gets its own reasoning back, within the answer and in later history alike.
    """

    text: str = ""
    raw: str = ""
    calls: dict[int, ToolCall] = field(default_factory=dict)

    def ordered(self) -> list[ToolCall]:
        return [self.calls[index] for index in sorted(self.calls)]

    def as_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.raw or None}
        calls = self.ordered()
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_args},
                    **({"extra_content": call.extra_content} if call.extra_content else {}),
                }
                for call in calls
            ]
        return message


def tool_message(call: ToolCall, result) -> dict:
    """One reply per tool call id; the protocol requires the content to be a string."""
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}


def _failure(call: ToolCall, text: str) -> tuple[dict, dict]:
    return tool_message(call, text), {"status": "error", "hits": [], "error": text}


async def run_tool(
    call: ToolCall, seen: Callable[[str], bool] | None = None
) -> tuple[dict, dict]:
    """Run one tool call, returning its protocol reply and the outcome the UI shows.

    Every failure becomes a tool reply instead of an exception: the model needs the error
    text to correct itself, and the protocol needs an answer for every call id. A tool with
    sources returns (reply, hits), and an empty hits list makes the outcome "empty".
    `seen` goes to the knowledge-base search, replacing any value the model sent.
    """
    args = call.args
    if args is None:
        return _failure(call, "Arguments were not a valid JSON object; call the tool again.")
    if call.name not in TOOLS or not args:
        return _failure(call, "Tool not found or no arguments provided or arguments are empty")
    fn = TOOLS[call.name]
    if call.name == "search_knowledge_base" and seen is not None:
        args = {**args, "seen": seen}
    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            result = await asyncio.to_thread(fn, **args)
    except ToolError as e:
        return _failure(call, str(e))
    except Exception as e:
        return _failure(call, f"Error running tool: {e}")
    summary, hits = result if isinstance(result, tuple) else (result, None)
    if summary is None:
        summary = "tool returned None"
    status = "empty" if hits == [] else "ok"
    return tool_message(call, summary), {"status": status, "hits": hits or []}


def _seen_in(messages: list[dict]) -> Callable[[str], bool]:
    """Whether a block of text already sits in a tool reply the model has.

    Taken once per round, so parallel calls of one round do not see each other: two
    searches finding the same chunk in one round both send it in full. That costs one
    duplicate chunk and keeps the prompt independent of which call finished first.
    """
    replies = "\n".join(m["content"] for m in messages if m.get("role") == "tool")
    return lambda block: block in replies


def page_images_message(
    pages: list[tuple[str, int]], shown: set[tuple[str, int]]
) -> dict | None:
    """The images of retrieved pages not yet shown in this answer, as one user message.

    A user message because the chat protocol takes images nowhere else: Google answers an
    image inside a tool reply with 400 "Invalid content part type: image_url" (1 Oct 2026).
    Each image is a page_image reference that to_wire renders, so the message can be stored
    and replayed without keeping image bytes.
    """
    parts: list[dict] = []
    for source, page in dict.fromkeys(pages):
        if (source, page) in shown:
            continue
        shown.add((source, page))
        parts += [
            {"type": "text", "text": f"{source}, page {page}:"},
            {"type": "page_image", "source": source, "page": page},
        ]
    if not parts:
        return None
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Page images of the knowledge-base results above, attached automatically "
                    "by the system, not written by the user. Use them for figures, tables and "
                    "anything the text transcript lost."
                ),
            },
            *parts,
        ],
    }


async def to_wire(messages: list[dict], rendered: dict[tuple[str, int], dict]) -> list[dict]:
    """The messages as the provider takes them: each page_image reference rendered to an image.

    `rendered` caches parts by (source, page) across the calls of one request, so a page sent
    in every round is rendered once. `page` numbers from 1, render_page from 0.
    """
    wire: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            wire.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") == "page_image":
                key = (part["source"], part["page"])
                if key not in rendered:
                    rendered[key] = await _render_part(*key)
                part = rendered[key]
            parts.append(part)
        wire.append({**message, "content": parts})
    return wire


async def _render_part(source: str, page: int) -> dict:
    # source comes back from stored history, so it must name a file in storage, not a path.
    if Path(source).name == source:
        try:
            path = Path(settings.storage_dir) / source
            return image_part(await asyncio.to_thread(render_page, str(path), page - 1))
        except Exception:
            logger.warning("Could not render page %d of %s", page, source, exc_info=True)
    return {"type": "text", "text": "[page image unavailable]"}


def _assistant_text(msg: dict) -> str:
    if msg.get("content"):
        return msg["content"]
    blocks = msg.get("blocks") or []
    parts = [b["text"] for b in blocks if b.get("type") == "answer" and b.get("text")]
    return "\n".join(parts)


def build_system_instruction() -> str:
    docs = vector_db.list_sources()
    base = settings.main_agent_prompt.strip()
    if not docs:
        return (
            f"{base}\n\n"
            "Knowledge base: empty. If the user asks about uploaded files, "
            "tell them to upload a document in the sidebar first."
        )
    listing = "\n".join(f"  - {name}" for name in docs)
    return (
        f"{base}\n\n"
        "Indexed documents in the knowledge base "
        "(use search_knowledge_base; pass filename to search within one file):\n"
        f"{listing}\n"
        "When the user asks to search RAG / the knowledge base, call search_knowledge_base "
        "with a concrete query derived from the conversation — do not ask what to search "
        "if the topic is already clear from chat history."
    )


def to_messages(history: list[dict] | None) -> list[dict]:
    """Convert stored chat history into chat messages. Empty messages are skipped.

    An answer with a stored context is replayed as it was sent, minus its thoughts.
    """
    messages: list[dict] = []
    for msg in history or []:
        if msg.get("context"):
            messages.extend(m for m in map(_without_thoughts, msg["context"]) if m)
            continue
        text = _assistant_text(msg)
        if not text.strip():
            continue
        role = "assistant" if msg.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": text})
    return messages


def _without_thoughts(message: dict) -> dict | None:
    """The message without earlier answers' thoughts, or None when nothing else is left.

    Temporary, while the answer model is Gemma 4 on the free tier (Luna, 1 Oct 2026): its
    model card says to strip them between user turns, and one 51k-character thinking loop
    kept in history made every later request larger than the 16 000 input-token per-minute
    quota. The database keeps them; per-provider handling comes with the provider branches.
    """
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str):
        return message
    # A thought cut off by max_tokens has no closing tag and runs to the end.
    content = re.sub(r"<thought>.*?(?:</thought>|$)", "", content, flags=re.S).strip() or None
    if content is None and not message.get("tool_calls"):
        return None
    return {**message, "content": content}


def is_retryable(e: Exception) -> bool:
    if isinstance(e, APIStatusError):
        return e.status_code in RETRYABLE_API_CODES
    return isinstance(e, (APIConnectionError, TimeoutError))


async def stream_turn(messages: list[dict], turn: Turn, use_tools: bool = True):
    """Stream one model turn, yielding UI events and filling `turn`.

    Tool calls arrive in pieces: `function.arguments` is a fragment of a JSON string and
    only `index` says which call it belongs to, so fragments are joined per index. Google's
    endpoint sends every call whole and without an index, so each of those is a call of its own.

    Gemma on Google's endpoint streams its thinking inside the content, wrapped in
    <thought>...</thought> (measured 25 Sep 2026); that part is a thought, not the answer.
    """
    request: dict = {
        "model": settings.main_model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 16000,
        "stream": True,
    }
    if use_tools:
        request["tools"] = TOOL_SCHEMAS
        request["tool_choice"] = "auto"

    text: list[str] = []
    raw: list[str] = []
    in_thought = False
    held = ""

    def content_event(piece: str) -> dict:
        if in_thought:
            return {"type": "thought_delta", "text": piece}
        text.append(piece)
        return {"type": "text_delta", "text": piece}

    stream = await client.chat.completions.create(**request)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Reasoning is outside the OpenAI spec: vLLM and DeepSeek send reasoning_content,
        # OpenRouter sends reasoning, and providers without it send neither.
        reasoning = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reasoning:
            yield {"type": "thought_delta", "text": reasoning}
        raw.append(delta.content or "")
        rest = held + (delta.content or "")
        held = ""
        while rest:
            marker = "</thought>" if in_thought else "<thought>"
            piece, tag, rest = rest.partition(marker)
            if not tag:
                # A marker can arrive cut across two chunks: hold back a tail that may start it.
                cut = next((n for n in range(len(marker) - 1, 0, -1) if piece.endswith(marker[:n])), 0)
                piece, held = piece[: len(piece) - cut], piece[len(piece) - cut :]
            if piece:
                yield content_event(piece)
            if tag:
                in_thought = not in_thought
        for fragment in delta.tool_calls or []:
            index = fragment.index if fragment.index is not None else len(turn.calls)
            call = turn.calls.setdefault(index, ToolCall())
            if fragment.id:
                call.id = fragment.id
            function = fragment.function
            if function and function.name:
                call.name += function.name
            if function and function.arguments:
                call.raw_args += function.arguments
            extra_content = getattr(fragment, "extra_content", None)
            if extra_content:
                call.extra_content = extra_content
    if held:
        yield content_event(held)
    turn.text = "".join(text)
    turn.raw = "".join(raw)


async def agent_loop(
    query: str, history: list[dict] | None = None, context: list[dict] | None = None
):
    """Stream one answer as UI events, running tool rounds until the model stops asking.

    `messages` is rebuilt on every attempt because the loop appends to it as it goes: a
    retry has to resend the original conversation, not the half-built one.

    On success `context` receives every message the model saw after the query, the answer
    included, for the caller to store and replay as history: the next request then starts
    with the same prefix.
    """
    for attempt in range(RETRIES):
        try:
            messages = [
                {"role": "system", "content": build_system_instruction()},
                *to_messages(history),
                {"role": "user", "content": query},
            ]
            start = len(messages)
            shown = {
                (part["source"], part["page"])
                for message in messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "page_image"
            }
            rendered: dict[tuple[str, int], dict] = {}
            turn = Turn()
            async for event in stream_turn(await to_wire(messages, rendered), turn):
                yield event

            iterations = 0
            while turn.ordered() and iterations < MAX_ITERATIONS:
                calls = turn.ordered()
                yield {
                    "type": "tool_start",
                    "name": [call.name for call in calls],
                    "args": [call.args or {} for call in calls],
                }
                messages.append(turn.as_message())
                # Each call reports the moment it finishes, so the UI learns which one by
                # index; the model still receives the replies in call order.
                seen = _seen_in(messages)
                pending = {
                    asyncio.create_task(run_tool(call, seen)): index
                    for index, call in enumerate(calls)
                }
                replies: dict[int, dict] = {}
                outcomes: dict[int, dict] = {}
                try:
                    while pending:
                        finished, _ = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in finished:
                            index = pending.pop(task)
                            replies[index], outcomes[index] = task.result()
                            yield {
                                "type": "tool_result",
                                "index": index,
                                "result": outcomes[index],
                            }
                finally:
                    # asyncio.wait leaves its tasks running when this generator is cancelled.
                    for task in pending:
                        task.cancel()
                messages.extend(replies[index] for index in range(len(calls)))
                if settings.answer_page_images:
                    pages = [
                        (hit["title"], page)
                        for index, call in enumerate(calls)
                        if call.name == "search_knowledge_base"
                        for hit in outcomes[index]["hits"]
                        for page in hit.get("pages", [])
                    ]
                    images = page_images_message(pages, shown)
                    if images:
                        messages.append(images)
                turn = Turn()
                async for event in stream_turn(await to_wire(messages, rendered), turn):
                    yield event
                iterations += 1

            if turn.ordered():
                messages.append(turn.as_message())
                # Every tool_call id must be answered before the next user message,
                # otherwise the provider rejects the whole request.
                messages.extend(
                    tool_message(call, "Tool budget exhausted, answer without this result.")
                    for call in turn.ordered()
                )
                messages.append({
                    "role": "user",
                    "content": "Give final answer based on what you get, you're out of tool calls",
                })
                turn = Turn()
                async for event in stream_turn(
                    await to_wire(messages, rendered), turn, use_tools=False
                ):
                    yield event

            if turn.raw:
                # A call in the final turn is never run, so storing it would leave an
                # unanswered tool_call id in the history and break every later request.
                final = turn.as_message()
                final.pop("tool_calls", None)
                messages.append(final)
            if context is not None:
                context[:] = messages[start:]
            yield {"type": "done"}
            return

        except Exception as e:
            yield {"type": "stream_reset"}

            if not is_retryable(e):
                logger.exception("Agent failed with a non-retryable error")
                yield {
                    'type': 'error',
                    'message': 'Something went wrong on our side. The request cannot be completed.'
                }
                return

            logger.warning("Agent attempt %d/%d failed: %s", attempt + 1, RETRIES, e)
            if attempt == RETRIES - 1:
                yield {
                    "type": "error",
                    'message': 'The model is unavailable right now, please try again in a moment.'
                }
                return
            await asyncio.sleep(min(BASE_DELAY * 2 ** attempt, MAX_DELAY))
