from types import SimpleNamespace

import pytest
from google.genai import types

import app.services.agent as agent_module
from app.services.agent import to_gemini_history, run_tool, TOOLS


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
    class FakeChat:
        async def send_message_stream(self, message):
            async def stream():
                yield SimpleNamespace(
                    parts=[types.Part(text="Проверяю вопрос", thought=True)]
                )
                yield SimpleNamespace(parts=[types.Part(text="Готовый ответ")])

            return stream()

    class FakeChats:
        def create(self, **kwargs):
            return FakeChat()

    fake_client = SimpleNamespace(aio=SimpleNamespace(chats=FakeChats()))

    monkeypatch.setattr(agent_module, "client", fake_client)
    monkeypatch.setattr(
        agent_module,
        "build_system_instruction",
        lambda: "Test system instruction",
    )

    events = [event async for event in agent_module.agent_loop("Тестовый вопрос")]
   
    assert events == [
        {"type": "thought_delta", "text": 'Проверяю вопрос'},
        {"type": "text_delta", "text": 'Готовый ответ'},
        {'type': 'done'}
    ]
