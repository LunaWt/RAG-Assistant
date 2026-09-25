import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field

from openai import APIConnectionError, APIStatusError

from app.config import settings
from app.services.llm_client import client
from app.services.tools import ToolError, calculator, search_knowledge_base, web_search
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
    """One model turn: the text it streamed and the tool calls it asked for."""

    text: str = ""
    calls: dict[int, ToolCall] = field(default_factory=dict)

    def ordered(self) -> list[ToolCall]:
        return [self.calls[index] for index in sorted(self.calls)]

    def as_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.text or None}
        calls = self.ordered()
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_args},
                }
                for call in calls
            ]
        return message


def tool_message(call: ToolCall, result) -> dict:
    """One reply per tool call id; the protocol requires the content to be a string."""
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}


def _failure(call: ToolCall, text: str) -> tuple[dict, dict]:
    return tool_message(call, text), {"status": "error", "hits": [], "error": text}


async def run_tool(call: ToolCall) -> tuple[dict, dict]:
    """Run one tool call, returning its protocol reply and the outcome the UI shows.

    Every failure becomes a tool reply instead of an exception: the model needs the error
    text to correct itself, and the protocol needs an answer for every call id. A tool with
    sources returns (reply, hits), and an empty hits list makes the outcome "empty".
    """
    args = call.args
    if args is None:
        return _failure(call, "Arguments were not a valid JSON object; call the tool again.")
    if call.name not in TOOLS or not args:
        return _failure(call, "Tool not found or no arguments provided or arguments are empty")
    fn = TOOLS[call.name]
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
    """Convert stored chat history into chat messages. Empty messages are skipped."""
    messages: list[dict] = []
    for msg in history or []:
        text = _assistant_text(msg)
        if not text.strip():
            continue
        role = "assistant" if msg.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": text})
    return messages


def is_retryable(e: Exception) -> bool:
    if isinstance(e, APIStatusError):
        return e.status_code in RETRYABLE_API_CODES
    return isinstance(e, (APIConnectionError, TimeoutError))


async def stream_turn(messages: list[dict], turn: Turn, use_tools: bool = True):
    """Stream one model turn, yielding UI events and filling `turn`.

    Tool calls arrive in pieces: `function.arguments` is a fragment of a JSON string and
    only `index` says which call it belongs to, so fragments are joined per index.
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
        if delta.content:
            text.append(delta.content)
            yield {"type": "text_delta", "text": delta.content}
        for fragment in delta.tool_calls or []:
            call = turn.calls.setdefault(fragment.index, ToolCall())
            if fragment.id:
                call.id = fragment.id
            function = fragment.function
            if function and function.name:
                call.name += function.name
            if function and function.arguments:
                call.raw_args += function.arguments
    turn.text = "".join(text)


async def agent_loop(query: str, history: list[dict] | None = None):
    """Stream one answer as UI events, running tool rounds until the model stops asking.

    `messages` is rebuilt on every attempt because the loop appends to it as it goes: a
    retry has to resend the original conversation, not the half-built one.
    """
    for attempt in range(RETRIES):
        try:
            messages = [
                {"role": "system", "content": build_system_instruction()},
                *to_messages(history),
                {"role": "user", "content": query},
            ]
            turn = Turn()
            async for event in stream_turn(messages, turn):
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
                pending = {
                    asyncio.create_task(run_tool(call)): index
                    for index, call in enumerate(calls)
                }
                replies: dict[int, dict] = {}
                try:
                    while pending:
                        finished, _ = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in finished:
                            index = pending.pop(task)
                            replies[index], outcome = task.result()
                            yield {"type": "tool_result", "index": index, "result": outcome}
                finally:
                    # asyncio.wait leaves its tasks running when this generator is cancelled.
                    for task in pending:
                        task.cancel()
                messages.extend(replies[index] for index in range(len(calls)))
                turn = Turn()
                async for event in stream_turn(messages, turn):
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
                async for event in stream_turn(messages, Turn(), use_tools=False):
                    yield event

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
