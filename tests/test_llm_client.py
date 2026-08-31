import pytest
from types import SimpleNamespace

import app.services.llm_client as llm_module


def fake_client(**models_attrs) -> SimpleNamespace:
    """Stand-in for genai.Client exposing only client.aio.models.<name>."""
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(**models_attrs)))


@pytest.mark.asyncio
async def test_generate_title_success(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_response() -> SimpleNamespace:
        return SimpleNamespace(text='  "Погода в Калифорнии"\n')

    async def generate_content(**kwargs) -> SimpleNamespace:
        return make_response()

    models = SimpleNamespace(generate_content=generate_content)
    aio = SimpleNamespace(models=models)
    fake_client = SimpleNamespace(aio=aio)

    monkeypatch.setattr(
        llm_module,
        'client',
        fake_client
    )

    title = await llm_module.generate_title('query')

    assert title == 'Погода в Калифорнии'


@pytest.mark.asyncio
async def test_generate_response_yields_only_nonempty_chunks(
    monkeypatch: pytest.MonkeyPatch
) -> None:

    async def generate_content_stream(**kwargs):
        async def generate_chunks():
            for text in ['Привет', None, ', мир']:
                yield SimpleNamespace(text=text)

        return generate_chunks()


    models = SimpleNamespace(generate_content_stream=generate_content_stream)
    aio = SimpleNamespace(models=models)
    fake_client = SimpleNamespace(aio=aio)

    monkeypatch.setattr(
        llm_module,
        'client',
        fake_client
    )
    resp = [c async for c in llm_module.generate_response('prompt')]

    assert resp == ['Привет', ', мир']


@pytest.mark.asyncio
async def test_generate_title_returns_empty_string_when_model_says_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank completion must not blow up: `or ""` keeps the caller's fallback reachable."""

    async def generate_content(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(text=None)

    monkeypatch.setattr(
        llm_module, "client", fake_client(generate_content=generate_content)
    )

    assert await llm_module.generate_title("query") == ""


@pytest.mark.asyncio
async def test_generate_response_yields_nothing_for_an_empty_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def generate_content_stream(**kwargs):
        async def chunks():
            return
            yield  # unreachable: makes this an async generator

        return chunks()

    monkeypatch.setattr(
        llm_module,
        "client",
        fake_client(generate_content_stream=generate_content_stream),
    )

    assert [c async for c in llm_module.generate_response("prompt")] == []


@pytest.mark.asyncio
async def test_generate_summary_returns_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def generate_content(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(text="Краткое содержание страниц")

    monkeypatch.setattr(
        llm_module, "client", fake_client(generate_content=generate_content)
    )

    assert await llm_module.generate_summary("long page text") == (
        "Краткое содержание страниц"
    )


@pytest.mark.asyncio
async def test_generate_summary_propagates_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must not swallow SDK failures — callers classify and retry them."""

    async def generate_content(**kwargs) -> SimpleNamespace:
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(
        llm_module, "client", fake_client(generate_content=generate_content)
    )

    with pytest.raises(RuntimeError, match="503"):
        await llm_module.generate_summary("long page text")