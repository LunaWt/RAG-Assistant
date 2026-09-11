import pytest
from types import SimpleNamespace

import app.services.llm_client as llm_module


def completion(content: str | None) -> SimpleNamespace:
    """One chat completion, shaped like the OpenAI response object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def fake_client(create) -> SimpleNamespace:
    """Stand-in for AsyncOpenAI exposing only client.chat.completions.create."""
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def install(monkeypatch: pytest.MonkeyPatch, create) -> list[dict]:
    """Capture the request kwargs so a test can assert what was actually sent."""
    sent: list[dict] = []

    async def recording_create(**kwargs):
        sent.append(kwargs)
        return await create(**kwargs)

    monkeypatch.setattr(llm_module, "client", fake_client(recording_create))
    return sent


@pytest.mark.asyncio
async def test_generate_title_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(**kwargs):
        return completion('  "Погода в Калифорнии"\n')

    sent = install(monkeypatch, create)

    assert await llm_module.generate_title("query") == "Погода в Калифорнии"
    assert sent[0]["model"] == llm_module.settings.summary_model
    assert sent[0]["messages"][-1] == {"role": "user", "content": "query"}


@pytest.mark.asyncio
async def test_generate_title_returns_empty_string_when_model_says_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank completion must not blow up: the caller's fallback stays reachable."""

    async def create(**kwargs):
        return completion(None)

    install(monkeypatch, create)

    assert await llm_module.generate_title("query") == ""


@pytest.mark.asyncio
async def test_generate_title_survives_a_response_without_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered or empty response has no choices at all; indexing [0] would raise."""

    async def create(**kwargs):
        return SimpleNamespace(choices=[])

    install(monkeypatch, create)

    assert await llm_module.generate_title("query") == ""


@pytest.mark.asyncio
async def test_generate_summary_returns_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create(**kwargs):
        return completion("Краткое содержание страниц")

    install(monkeypatch, create)

    assert await llm_module.generate_summary("long page text") == (
        "Краткое содержание страниц"
    )


@pytest.mark.asyncio
async def test_generate_summary_propagates_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must not swallow SDK failures — callers classify and retry them."""

    async def create(**kwargs):
        raise RuntimeError("503 Service Unavailable")

    install(monkeypatch, create)

    with pytest.raises(RuntimeError, match="503"):
        await llm_module.generate_summary("long page text")


def test_client_does_not_retry_on_its_own() -> None:
    """agent_loop runs five attempts; SDK-level retries would silently multiply them."""
    assert llm_module.client.max_retries == 0
