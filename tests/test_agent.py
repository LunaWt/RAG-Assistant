from typing import Any

import pytest
from google.genai import types
from google.genai.errors import APIError
from httpx import TransportError

import app.services.agent as agent_module
from app.services.agent import to_gemini_history, run_tool, TOOLS
from tests.fakes import (
    LoopingChat,
    ScriptedChat,
    call_part,
    chunk,
    scripted_client,
    text_part,
)


def install_chat(monkeypatch: pytest.MonkeyPatch, chat: Any) -> None:
    """Point agent_loop at a scripted chat and a fixed system instruction."""
    monkeypatch.setattr(agent_module, "client", scripted_client(chat))
    monkeypatch.setattr(
        agent_module, "build_system_instruction", lambda: "Test system instruction"
    )


@pytest.fixture
def fake_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps = []
    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(
        agent_module.asyncio, 
        'sleep', 
        fake_sleep
    )
    return sleeps


@pytest.mark.asyncio
async def test_run_tool_unknown_tool_name():
    fc = types.FunctionCall(name="nonexistent_tool", args={"query": "hi"})

    part, hits = await run_tool(fc)

    assert part.function_response.name == "nonexistent_tool"
    assert "Tool not found" in part.function_response.response["result"]
    assert hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{}, None])
async def test_run_tool_empty_args(args):
    fc = types.FunctionCall(name="calculator", args=args)

    part, hits = await run_tool(fc)

    assert (
        "Tool not found or no arguments provided"
        in part.function_response.response["result"]
    )
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_wrong_arg_names():
    fc = types.FunctionCall(name="calculator", args={"foo": "2 + 2"})

    part, hits = await run_tool(fc)

    assert part.function_response.response["result"].startswith("Error running tool")
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_sync_tool_success():
    fc = types.FunctionCall(name="calculator", args={"expression": "2 + 2"})

    part, hits = await run_tool(fc)

    assert part.function_response.name == "calculator"
    assert part.function_response.response["result"] == 4
    assert hits == []


@pytest.mark.asyncio
async def test_run_tool_zero_is_a_valid_result():
    fc = types.FunctionCall(name="calculator", args={"expression": "2 - 2"})

    part, hits = await run_tool(fc)

    assert part.function_response.response["result"] == 0


@pytest.mark.asyncio
async def test_run_tool_async_tool_success(monkeypatch: pytest.MonkeyPatch):
    async def fake_web_search(query: str) -> tuple[str, list[dict]]:
        return "В Калифорнии сегодня солнечно!", [{"query": query}]

    monkeypatch.setitem(TOOLS, "web_search", fake_web_search)

    fc = types.FunctionCall(
        name="web_search", args={"query": "Какая погода в калифорнии?"}
    )
    part, hits = await run_tool(fc)

    assert part.function_response.response["result"] == "В Калифорнии сегодня солнечно!"
    assert hits == [{"query": "Какая погода в калифорнии?"}]


def test_maps_user_and_assistant_roles():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = to_gemini_history(history)

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].parts[0].text == "hi"
    assert result[1].role == "model"  # assistant -> model
    assert result[1].parts[0].text == "hello"


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
    result = to_gemini_history(history)

    assert len(result) == 1
    assert result[0].role == "model"
    assert result[0].parts[0].text == "final answer"


def test_skips_blank_messages():
    history = [
        {"role": "user", "content": "real"},
        {"role": "assistant", "content": "   "},  # whitespace only -> skipped
        {"role": "user", "content": ""},  # empty -> skipped
    ]
    result = to_gemini_history(history)

    assert len(result) == 1
    assert result[0].parts[0].text == "real"


def test_empty_history_returns_empty_list():
    assert to_gemini_history([]) == []
    assert to_gemini_history(None) == []


@pytest.mark.asyncio
async def test_agent_loop_happy_path(monkeypatch: pytest.MonkeyPatch):
    # One turn, two chunks: the model streams a thought and then the answer.
    install_chat(
        monkeypatch,
        ScriptedChat(
            [
                [
                    chunk(text_part("Проверяю вопрос", thought=True)),
                    chunk(text_part("Готовый ответ")),
                ]
            ]
        ),
    )

    events = [event async for event in agent_module.agent_loop("Тестовый вопрос")]
   
    assert events == [
        {"type": "thought_delta", "text": 'Проверяю вопрос'},
        {"type": "text_delta", "text": 'Готовый ответ'},
        {'type': 'done'}
    ]


@pytest.mark.asyncio
async def test_agent_loop_tool_turn(monkeypatch: pytest.MonkeyPatch):
    fake_chat = ScriptedChat(
        turns=[
            [chunk(
                text_part('I need to call "calculator"', thought=True),
                call_part('calculator', expression='2+2'),
                text_part('Сейчас посмотрю'),
            )],
            [chunk(
                text_part('I have the answer!', thought=True),
                text_part('4'),
            )],
        ]
    )
    install_chat(monkeypatch, fake_chat)

    events = [event async for event in agent_module.agent_loop("Тестовый вопрос")]
       
    assert events == [
        {
            "type": "thought_delta", 
            "text": 'I need to call "calculator"'
        },
        {
            'type': 'text_delta',
            'text': 'Сейчас посмотрю'
        },
        {
            'type': 'tool_start',
            'name': ['calculator'],
            'args':[{'expression': '2+2'}],
        },
        {
            'type': 'thought_delta',
            'text': 'I have the answer!'
        },
        {
            'type': 'text_delta',
            'text': '4'
        },
        {
            'type': 'done'
        }
    ]
    assert fake_chat.sent[1][0].function_response.response == {'result': 4}


@pytest.mark.asyncio
async def test_agent_loop_emits_tool_hits_from_an_async_tool(
    monkeypatch: pytest.MonkeyPatch,
):
    """Search results reach the UI as their own event, separate from tool_start."""
    hits = [{"title": "Weather in California", "href": "https://example.com/ca"}]

    async def fake_web_search(query: str) -> tuple[str, list[dict]]:
        return "Сегодня солнечно", hits

    monkeypatch.setitem(TOOLS, "web_search", fake_web_search)

    fake_chat = ScriptedChat(
        turns=[
            [chunk(call_part("web_search", query="погода в Калифорнии"))],
            [chunk(text_part("В Калифорнии солнечно"))],
        ]
    )
    install_chat(monkeypatch, fake_chat)

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
    assert fake_chat.sent[1][0].function_response.response == {
        "result": "Сегодня солнечно"
    }


@pytest.mark.asyncio
async def test_agent_loop_runs_two_tool_rounds_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
):
    """The model may keep calling tools; every round gets its own tool_start."""
    fake_chat = ScriptedChat(
        turns=[
            [chunk(call_part("calculator", expression="2 + 2"))],
            [chunk(call_part("calculator", expression="4 * 10"))],
            [chunk(text_part("Итого 40"))],
        ]
    )
    install_chat(monkeypatch, fake_chat)

    events = [event async for event in agent_module.agent_loop("Посчитай")]

    assert events == [
        {"type": "tool_start", "name": ["calculator"], "args": [{"expression": "2 + 2"}]},
        {"type": "tool_start", "name": ["calculator"], "args": [{"expression": "4 * 10"}]},
        {"type": "text_delta", "text": "Итого 40"},
        {"type": "done"},
    ]
    # Each round's real result must reach the model before the next turn.
    assert fake_chat.sent[1][0].function_response.response == {"result": 4}
    assert fake_chat.sent[2][0].function_response.response == {"result": 40}


@pytest.mark.asyncio
async def test_agent_loop_runs_parallel_calls_of_one_turn_together(
    monkeypatch: pytest.MonkeyPatch,
):
    """Several calls in one turn produce one tool_start and one batch of results."""
    fake_chat = ScriptedChat(
        turns=[
            [chunk(
                call_part("calculator", expression="2 + 2"),
                call_part("calculator", expression="10 / 4"),
            )],
            [chunk(text_part("4 и 2.5"))],
        ]
    )
    install_chat(monkeypatch, fake_chat)

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
    results = [p.function_response.response["result"] for p in fake_chat.sent[1]]
    assert results == [4, 2.5]  # gather() must preserve call order


@pytest.mark.asyncio
async def test_agent_loop_finishes_on_an_empty_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    """A model turn with no parts still terminates with done and nothing else."""
    install_chat(monkeypatch, ScriptedChat(turns=[[]]))

    events = [event async for event in agent_module.agent_loop("Тишина")]

    assert events == [{"type": "done"}]


@pytest.mark.asyncio
async def test_agent_does_not_retry_programming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  
    class BrokenChat:
        def __init__(self):
            self.calls = 0

        async def send_message_stream(self, message):
            self.calls += 1
            raise TypeError("'NoneType' object is not iterable")

    chat = BrokenChat()
    install_chat(monkeypatch, chat)

    events = [event async for event in agent_module.agent_loop("Test TypeError")]

    assert chat.calls == 1
    assert [e["type"] for e in events] == ['stream_reset', 'error']
    assert events[1]['message'] == 'Something went wrong on our side. The request cannot be completed.'


@pytest.mark.asyncio
async def test_agent_ends_on_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_sleeps: list[float]
) -> None:
    class BrokenChat:
        def __init__(self):
            self.calls = 0

        async def send_message_stream(self, message):
            self.calls += 1
            raise APIError(408, {"error": {"message": "x", "status": "TIMEOUT"}})
    
    chat = BrokenChat()
    install_chat(monkeypatch, chat)
    
    events = [event async for event in agent_module.agent_loop("Test retries")]

    assert chat.calls == agent_module.RETRIES
    assert len(events) == agent_module.RETRIES + 1
    assert events[-1]['message'] == 'The model is unavailable right now, please try again in a moment.'
    assert fake_sleeps == [1.0, 2.0, 4.0, 8.0]


def test_is_retryable() -> None:
    assert agent_module.is_retryable(TypeError("boom")) is False
    assert agent_module.is_retryable(TransportError('connection reset by peer')) is True
    assert agent_module.is_retryable(APIError(
        503, 
        {"error": {"message": "x", "status": "BAD"}})) is True
    assert agent_module.is_retryable(APIError(
        400, 
        {"error": {"message": "x", "status": "BAD"}})) is False
    assert agent_module.is_retryable(APIError(
        429, 
        {"error": {"message": "x", "status": "BAD"}})) is True


@pytest.mark.asyncio
async def test_agent_successful_retry_after_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_sleeps: list[float]
) -> None:
    class BrokenChat:
        def __init__(self):
            self.chunk = chunk(text_part('Answer'))
            self.calls = 0

        async def send_message_stream(self, message):
            self.calls += 1
            if self.calls < 3:
                raise APIError(408, {"error": {"message": "x", "status": "TIMEOUT"}})
            
            async def stream():
                yield self.chunk

            return stream()

    chat = BrokenChat()
    install_chat(monkeypatch, chat)
    
    events = [event async for event in agent_module.agent_loop("Test answer after retries")]

    assert events == [
        {"type": "stream_reset"}, 
        {"type": "stream_reset"}, 
        {"type": "text_delta", "text": "Answer"},
        {"type": "done"}
    ]
    assert fake_sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_agent_loop_stops_at_the_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that never stops calling tools is cut off at exactly MAX_ITERATIONS rounds."""
    chat = LoopingChat(call_part("calculator", expression="2 + 2"))
    install_chat(monkeypatch, chat)

    events = [event async for event in agent_module.agent_loop("Считай без остановки")]

    tool_rounds = [event for event in events if event["type"] == "tool_start"]
    assert len(tool_rounds) == agent_module.MAX_ITERATIONS
    # Opening turn + one per tool round + the final "you're out of tool calls" nudge.
    assert chat.calls == agent_module.MAX_ITERATIONS + 2
    assert events[-1] == {"type": "done"}