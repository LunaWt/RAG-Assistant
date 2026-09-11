import copy
import json
from typing import Any

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError

import app.services.agent as agent_module
from app.services.agent import TOOLS, ToolCall, run_tool, to_messages
from tests.fakes import (
    LoopingClient,
    ScriptedClient,
    call_chunk,
    call_fragment,
    chunk,
    text_chunk,
    thought_chunk,
)

NUDGE = "Give final answer based on what you get, you're out of tool calls"


def install_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    """Point agent_loop at a scripted client and a fixed system instruction."""
    monkeypatch.setattr(agent_module, "client", client)
    monkeypatch.setattr(
        agent_module, "build_system_instruction", lambda: "Test system instruction"
    )


def call(name: str, index: int = 0, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{index}_{name}", name=name, raw_args=json.dumps(args))


def api_error(status: int) -> APIStatusError:
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    return APIStatusError(
        "boom", response=httpx2.Response(status, request=request), body=None
    )


@pytest.fixture
def fake_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(agent_module.asyncio, "sleep", fake_sleep)
    return sleeps


@pytest.mark.asyncio
async def test_run_tool_unknown_tool_name():
    message, hits = await run_tool(call("nonexistent_tool", query="hi"))

    assert message["tool_call_id"] == "call_0_nonexistent_tool"
    assert "Tool not found" in message["content"]
    assert hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_args", ["{}", ""])
async def test_run_tool_empty_args(raw_args):
    message, hits = await run_tool(
        ToolCall(id="c0", name="calculator", raw_args=raw_args)
    )

    assert "Tool not found or no arguments provided" in message["content"]
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_reports_broken_json_instead_of_crashing():
    """A truncated argument stream must come back as a message the model can act on."""
    message, hits = await run_tool(
        ToolCall(id="c0", name="calculator", raw_args='{"expression": "2 +')
    )

    assert "not a valid JSON object" in message["content"]
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_wrong_arg_names():
    message, hits = await run_tool(call("calculator", foo="2 + 2"))

    assert message["content"].startswith("Error running tool")
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_sync_tool_success():
    message, hits = await run_tool(call("calculator", expression="2 + 2"))

    assert message == {
        "role": "tool",
        "tool_call_id": "call_0_calculator",
        "content": "4",
    }
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_zero_is_a_valid_result():
    message, _ = await run_tool(call("calculator", expression="2 - 2"))

    assert message["content"] == "0"


@pytest.mark.asyncio
async def test_run_tool_async_tool_success(monkeypatch: pytest.MonkeyPatch):
    async def fake_web_search(query: str) -> tuple[str, list[dict]]:
        return "В Калифорнии сегодня солнечно!", [{"query": query}]

    monkeypatch.setitem(TOOLS, "web_search", fake_web_search)

    message, hits = await run_tool(
        call("web_search", query="Какая погода в калифорнии?")
    )

    assert message["content"] == "В Калифорнии сегодня солнечно!"
    assert hits == [{"query": "Какая погода в калифорнии?"}]


def test_maps_user_and_assistant_roles():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    assert to_messages(history) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_falls_back_to_answer_blocks():
    history = [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thought", "text": "thinking..."},  # not an answer -> ignored
                {"type": "answer", "text": "final answer"},  # used
            ],
        },
    ]

    assert to_messages(history) == [{"role": "assistant", "content": "final answer"}]


def test_skips_blank_messages():
    history = [
        {"role": "user", "content": "real"},
        {"role": "assistant", "content": "   "},  # whitespace only -> skipped
        {"role": "user", "content": ""},  # empty -> skipped
    ]

    assert to_messages(history) == [{"role": "user", "content": "real"}]


def test_empty_history_returns_empty_list():
    assert to_messages([]) == []
    assert to_messages(None) == []


@pytest.mark.asyncio
async def test_agent_loop_happy_path(monkeypatch: pytest.MonkeyPatch):
    client = ScriptedClient([[thought_chunk("Проверяю вопрос"), text_chunk("Готовый ответ")]])
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Тестовый вопрос")]

    assert events == [
        {"type": "thought_delta", "text": "Проверяю вопрос"},
        {"type": "text_delta", "text": "Готовый ответ"},
        {"type": "done"},
    ]
    assert client.sent[0] == [
        {"role": "system", "content": "Test system instruction"},
        {"role": "user", "content": "Тестовый вопрос"},
    ]


@pytest.mark.asyncio
async def test_agent_loop_sends_history_between_system_and_query(
    monkeypatch: pytest.MonkeyPatch,
):
    client = ScriptedClient([[text_chunk("Ответ")]])
    install_client(monkeypatch, client)

    history = [{"role": "user", "content": "первый"}, {"role": "assistant", "content": "ответ"}]
    [event async for event in agent_module.agent_loop("второй", history=history)]

    assert [m["role"] for m in client.sent[0]] == ["system", "user", "assistant", "user"]
    assert client.sent[0][-1]["content"] == "второй"


@pytest.mark.asyncio
async def test_agent_loop_tool_turn(monkeypatch: pytest.MonkeyPatch):
    client = ScriptedClient(
        turns=[
            [
                thought_chunk('I need to call "calculator"'),
                call_chunk("calculator", expression="2+2"),
                text_chunk("Сейчас посмотрю"),
            ],
            [thought_chunk("I have the answer!"), text_chunk("4")],
        ]
    )
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Тестовый вопрос")]

    assert events == [
        {"type": "thought_delta", "text": 'I need to call "calculator"'},
        {"type": "text_delta", "text": "Сейчас посмотрю"},
        {
            "type": "tool_start",
            "name": ["calculator"],
            "args": [{"expression": "2+2"}],
        },
        {"type": "thought_delta", "text": "I have the answer!"},
        {"type": "text_delta", "text": "4"},
        {"type": "done"},
    ]
    assert client.tool_replies(1) == [
        {"role": "tool", "tool_call_id": "call_0_calculator", "content": "4"}
    ]


@pytest.mark.asyncio
async def test_agent_loop_emits_tool_hits_from_an_async_tool(
    monkeypatch: pytest.MonkeyPatch,
):
    """Search results reach the UI as their own event, separate from tool_start."""
    hits = [{"title": "Weather in California", "href": "https://example.com/ca"}]

    async def fake_web_search(query: str) -> tuple[str, list[dict]]:
        return "Сегодня солнечно", hits

    monkeypatch.setitem(TOOLS, "web_search", fake_web_search)

    client = ScriptedClient(
        turns=[
            [call_chunk("web_search", query="погода в Калифорнии")],
            [text_chunk("В Калифорнии солнечно")],
        ]
    )
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Какая погода?")]

    assert events == [
        {
            "type": "tool_start",
            "name": ["web_search"],
            "args": [{"query": "погода в Калифорнии"}],
        },
        {"type": "tool_hits", "query": "погода в Калифорнии", "hits": hits},
        {"type": "text_delta", "text": "В Калифорнии солнечно"},
        {"type": "done"},
    ]
    assert client.tool_replies(1)[0]["content"] == "Сегодня солнечно"


@pytest.mark.asyncio
async def test_agent_loop_runs_two_tool_rounds_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
):
    """The model may keep calling tools; every round gets its own tool_start."""
    client = ScriptedClient(
        turns=[
            [call_chunk("calculator", expression="2 + 2")],
            [call_chunk("calculator", expression="4 * 10")],
            [text_chunk("Итого 40")],
        ]
    )
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Посчитай")]

    assert events == [
        {"type": "tool_start", "name": ["calculator"], "args": [{"expression": "2 + 2"}]},
        {"type": "tool_start", "name": ["calculator"], "args": [{"expression": "4 * 10"}]},
        {"type": "text_delta", "text": "Итого 40"},
        {"type": "done"},
    ]
    # Each round's real result must reach the model before the next turn.
    assert [m["content"] for m in client.tool_replies(1)] == ["4"]
    assert [m["content"] for m in client.tool_replies(2)] == ["4", "40"]


@pytest.mark.asyncio
async def test_parallel_calls_are_answered_by_id(monkeypatch: pytest.MonkeyPatch):
    """Two calls to the same tool are told apart by id, not by tool name."""
    client = ScriptedClient(
        turns=[
            [
                call_chunk("calculator", index=0, expression="2 + 2"),
                call_chunk("calculator", index=1, expression="10 / 4"),
            ],
            [text_chunk("4 и 2.5")],
        ]
    )
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Посчитай оба")]

    assert events == [
        {
            "type": "tool_start",
            "name": ["calculator", "calculator"],
            "args": [{"expression": "2 + 2"}, {"expression": "10 / 4"}],
        },
        {"type": "text_delta", "text": "4 и 2.5"},
        {"type": "done"},
    ]
    assert client.tool_replies(1) == [
        {"role": "tool", "tool_call_id": "call_0_calculator", "content": "4"},
        {"role": "tool", "tool_call_id": "call_1_calculator", "content": "2.5"},
    ]


@pytest.mark.asyncio
async def test_tool_call_arguments_are_joined_by_index(monkeypatch: pytest.MonkeyPatch):
    """Arguments arrive as string fragments of two interleaved calls; only index separates them."""
    client = ScriptedClient(
        turns=[
            [
                chunk(tool_calls=[
                    call_fragment(0, id="a", name="calcu", arguments='{"expression": "2'),
                ]),
                chunk(tool_calls=[
                    call_fragment(1, id="b", name="calcu", arguments='{"expression": "10'),
                ]),
                chunk(tool_calls=[
                    call_fragment(0, name="lator", arguments=' + 2"}'),
                    call_fragment(1, name="lator", arguments=' / 4"}'),
                ]),
            ],
            [text_chunk("готово")],
        ]
    )
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Посчитай оба")]

    assert events[0] == {
        "type": "tool_start",
        "name": ["calculator", "calculator"],
        "args": [{"expression": "2 + 2"}, {"expression": "10 / 4"}],
    }
    assert client.tool_replies(1) == [
        {"role": "tool", "tool_call_id": "a", "content": "4"},
        {"role": "tool", "tool_call_id": "b", "content": "2.5"},
    ]


@pytest.mark.asyncio
async def test_agent_loop_finishes_on_an_empty_stream(monkeypatch: pytest.MonkeyPatch):
    """A model turn with no chunks still terminates with done and nothing else."""
    install_client(monkeypatch, ScriptedClient(turns=[[]]))

    events = [event async for event in agent_module.agent_loop("Тишина")]

    assert events == [{"type": "done"}]


@pytest.mark.asyncio
async def test_agent_does_not_retry_programming_error(monkeypatch: pytest.MonkeyPatch):
    class BrokenClient(ScriptedClient):
        async def _create(self, **kwargs):
            self.sent.append(kwargs["messages"])
            raise TypeError("'NoneType' object is not iterable")

    client = BrokenClient(turns=[])
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Test TypeError")]

    assert client.calls == 1
    assert [e["type"] for e in events] == ["stream_reset", "error"]
    assert events[1]["message"] == (
        "Something went wrong on our side. The request cannot be completed."
    )


@pytest.mark.asyncio
async def test_agent_ends_on_retryable_errors(
    monkeypatch: pytest.MonkeyPatch, fake_sleeps: list[float]
):
    class BrokenClient(ScriptedClient):
        async def _create(self, **kwargs):
            self.sent.append(kwargs["messages"])
            raise api_error(408)

    client = BrokenClient(turns=[])
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Test retries")]

    assert client.calls == agent_module.RETRIES
    assert len(events) == agent_module.RETRIES + 1
    assert events[-1]["message"] == (
        "The model is unavailable right now, please try again in a moment."
    )
    assert fake_sleeps == [1.0, 2.0, 4.0, 8.0]


def test_is_retryable():
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")

    assert agent_module.is_retryable(TypeError("boom")) is False
    assert agent_module.is_retryable(APIConnectionError(request=request)) is True
    assert agent_module.is_retryable(api_error(503)) is True
    assert agent_module.is_retryable(api_error(429)) is True
    assert agent_module.is_retryable(api_error(400)) is False


@pytest.mark.asyncio
async def test_agent_successful_retry_after_errors(
    monkeypatch: pytest.MonkeyPatch, fake_sleeps: list[float]
):
    class FlakyClient(ScriptedClient):
        attempts = 0

        async def _create(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise api_error(408)
            return await super()._create(**kwargs)

    client = FlakyClient(turns=[[text_chunk("Answer")]])
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Test answer after retries")]

    assert events == [
        {"type": "stream_reset"},
        {"type": "stream_reset"},
        {"type": "text_delta", "text": "Answer"},
        {"type": "done"},
    ]
    assert fake_sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retry_restarts_from_the_original_messages(
    monkeypatch: pytest.MonkeyPatch, fake_sleeps: list[float]
):
    """A retry resends the original conversation, not the half-built one.

    agent_loop appends the assistant turn and the tool replies into `messages` as it goes,
    so the list is only safe to reuse if it is rebuilt inside the retry loop. The other
    retry tests fail on the very first model call, before any append, and cannot see this.
    """

    class FailsAfterFirstToolRound(ScriptedClient):
        attempts = 0

        async def _create(self, **kwargs):
            self.attempts += 1
            if self.attempts == 2:
                self.sent.append(copy.deepcopy(kwargs["messages"]))
                raise api_error(503)
            return await super()._create(**kwargs)

    client = FailsAfterFirstToolRound(
        turns=[
            [call_chunk("calculator", expression="2 + 2")],
            [call_chunk("calculator", expression="2 + 2")],
            [text_chunk("4")],
        ]
    )
    install_client(monkeypatch, client)
    history = [{"role": "user", "content": "раньше"}]

    events = [event async for event in agent_module.agent_loop("Посчитай", history=history)]

    # The failing call really did carry the half-built conversation...
    assert [m["role"] for m in client.sent[1]] == ["system", "user", "user", "assistant", "tool"]
    # ...and the retry went back to exactly what the first call sent.
    assert client.sent[2] == client.sent[0]
    assert events[-1] == {"type": "done"}
    assert fake_sleeps == [1.0]


@pytest.mark.asyncio
async def test_agent_loop_stops_at_the_iteration_limit(monkeypatch: pytest.MonkeyPatch):
    """A model that never stops calling tools is cut off at exactly MAX_ITERATIONS rounds."""
    client = LoopingClient(call_chunk("calculator", expression="2 + 2"))
    install_client(monkeypatch, client)

    events = [event async for event in agent_module.agent_loop("Считай без остановки")]

    tool_rounds = [event for event in events if event["type"] == "tool_start"]
    assert len(tool_rounds) == agent_module.MAX_ITERATIONS
    # Opening turn + one per tool round + the final "you're out of tool calls" nudge.
    assert client.calls == agent_module.MAX_ITERATIONS + 2
    assert events[-1] == {"type": "done"}


@pytest.mark.asyncio
async def test_iteration_limit_answers_every_pending_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """The nudge follows an assistant turn with tool_calls, which the protocol says
    must be answered first; an unanswered id makes the provider reject the request."""
    client = LoopingClient(
        call_chunk("calculator", index=0, expression="2 + 2"),
        call_chunk("calculator", index=1, expression="10 / 4"),
    )
    install_client(monkeypatch, client)

    [event async for event in agent_module.agent_loop("Считай без остановки")]

    nudge = client.sent[-1]
    # Only replies that come after the last assistant turn count: every round reuses
    # the same call ids, so scanning the whole list would find round 1's replies.
    last = max(i for i, m in enumerate(nudge) if m.get("tool_calls"))
    pending = {c["id"] for c in nudge[last]["tool_calls"]}
    answered = {m["tool_call_id"] for m in nudge[last + 1:] if m["role"] == "tool"}

    assert pending and pending <= answered
    assert nudge[-1] == {"role": "user", "content": NUDGE}
    assert client.requests[-1].get("tools") is None
