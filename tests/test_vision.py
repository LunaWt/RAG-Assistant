from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, NotFoundError

import app.services.vision as vision
from app.config import settings

SEP = settings.vision_page_separator


def response(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def transient() -> APIConnectionError:
    return APIConnectionError(request=httpx2.Request("POST", "https://vision.test"))


class FakeVisionClient:
    """Replays one prepared reply per request; an exception in the script is raised instead."""

    def __init__(self, *replies) -> None:
        self._replies = list(replies)
        self.batch_sizes: list[int] = []
        self.models: list[str] = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        self.batch_sizes.append(sum(1 for p in content if p["type"] == "image_url"))
        self.models.append(kwargs["model"])
        if not self._replies:
            raise AssertionError("unscripted vision request")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff is a delay, not behaviour: record what it would have waited and return at once."""
    waited: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(vision.asyncio, "sleep", fake_sleep)
    return waited


@pytest.fixture(autouse=True)
def two_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "vision_attempts", 2)
    monkeypatch.setattr(settings, "vision_model", "primary")
    monkeypatch.setattr(settings, "vision_fallback_model", "fallback")


@pytest.mark.asyncio
async def test_batch_splits_into_one_entry_per_page() -> None:
    client = FakeVisionClient(response(f"{SEP}\none\n{SEP}\ntwo"))

    assert await vision.pages_to_markdown(client, [b"a", b"b"]) == ["one", "two"]
    assert client.batch_sizes == [2]
    assert client.models == ["primary"]


@pytest.mark.asyncio
async def test_truncated_single_page_raises() -> None:
    """One page that will not fit the output budget is a real failure, not a retry case."""
    client = FakeVisionClient(response(f"{SEP}\none", finish_reason="length"))

    with pytest.raises(ValueError, match="output limit"):
        await vision.pages_to_markdown(client, [b"a"])


@pytest.mark.asyncio
async def test_single_page_mismatch_is_not_retried_forever() -> None:
    client = FakeVisionClient(response(f"{SEP}\na\n{SEP}\nb"))

    with pytest.raises(vision.PageCountMismatch):
        await vision.pages_to_markdown(client, [b"a"])
    assert client.batch_sizes == [1]


@pytest.mark.asyncio
async def test_transient_error_is_retried_on_the_same_model(
    no_sleeping: list[float],
) -> None:
    client = FakeVisionClient(transient(), response(f"{SEP}\none"))

    assert await vision.pages_to_markdown(client, [b"a"]) == ["one"]
    assert client.models == ["primary", "primary"]
    assert no_sleeping == [settings.vision_retry_backoff]


@pytest.mark.asyncio
async def test_exhausted_primary_falls_through_to_the_fallback_model() -> None:
    client = FakeVisionClient(transient(), transient(), response(f"{SEP}\none"))

    assert await vision.pages_to_markdown(client, [b"a"]) == ["one"]
    assert client.models == ["primary", "primary", "fallback"]


@pytest.mark.asyncio
async def test_no_sleep_after_the_last_attempt_of_the_last_model(
    no_sleeping: list[float],
) -> None:
    """Four attempts, three gaps: waiting after the final one only delays the failure."""
    client = FakeVisionClient(*[transient()] * 4)

    with pytest.raises(APIConnectionError):
        await vision.pages_to_markdown(client, [b"a"])
    assert len(client.models) == 4
    assert len(no_sleeping) == 2


def test_retry_after_header_overrides_backoff_and_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "vision_retry_max_sleep", 30.0)
    with_header = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "7"}))
    absurd = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "3600"}))

    assert vision._retry_delay(with_header, attempt=0) == 7.0
    assert vision._retry_delay(absurd, attempt=0) == 30.0
    assert vision._retry_delay(transient(), attempt=2) == 8.0


@pytest.mark.asyncio
async def test_truncated_batch_falls_back_to_one_call_per_page() -> None:
    """A batch overflowing the output cap is the case splitting it actually fixes."""
    client = FakeVisionClient(
        response(f"{SEP}\ncut off here", finish_reason="length"),
        response(f"{SEP}\none"),
        response(f"{SEP}\ntwo"),
    )

    pages = await vision._transcribe_batch(client, [b"a", b"b"], 1, _unexpected_failure)

    assert pages == ["one", "two"]
    assert client.batch_sizes == [2, 1, 1]


@pytest.mark.asyncio
async def test_short_batch_falls_back_to_one_call_per_page() -> None:
    """Two pages in, one page back: the batch is retried page by page rather than accepted."""
    client = FakeVisionClient(
        response(f"{SEP}\nonly one"),
        response(f"{SEP}\none"),
        response(f"{SEP}\ntwo"),
    )

    pages = await vision._transcribe_batch(client, [b"a", b"b"], 1, _unexpected_failure)

    assert pages == ["one", "two"]
    assert client.batch_sizes == [2, 1, 1]


@pytest.mark.asyncio
async def test_one_dead_page_does_not_take_its_batch_with_it() -> None:
    """The batch fails, the split runs, and only the page that keeps failing is lost."""
    client = FakeVisionClient(
        *[transient()] * 4,
        response(f"{SEP}\none"),
        *[transient()] * 4,
    )
    failed: list[int] = []

    pages = await vision._transcribe_batch(client, [b"a", b"b"], 7, failed.append)

    assert pages == ["one", "[page 8 could not be transcribed]"]
    assert failed == [8]


def _unexpected_failure(number: int) -> None:
    raise AssertionError(f"page {number} should not have failed")


def stub_pdf(monkeypatch: pytest.MonkeyPatch, pages: int) -> None:
    monkeypatch.setattr(vision, "page_count", lambda path: pages)
    monkeypatch.setattr(
        vision, "render_page", lambda path, number: f"png{number}".encode()
    )


def test_pdf_to_markdown_batches_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "vision_batch_pages", 2)
    stub_pdf(monkeypatch, 3)
    client = FakeVisionClient(
        response(f"{SEP}\none\n{SEP}\ntwo"), response(f"{SEP}\nthree")
    )
    monkeypatch.setattr(vision, "build_client", lambda: client)
    seen: list[tuple[int, int]] = []

    text = vision.pdf_to_markdown(
        "document.pdf", on_progress=lambda d, t: seen.append((d, t))
    )

    assert text == "one\n\ntwo\n\nthree"
    assert client.batch_sizes == [2, 1]
    assert seen == [(0, 3), (2, 3), (3, 3)]
    assert client.closed


def test_a_failed_page_becomes_a_marker_in_the_indexed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One page of three may fail at this ceiling, and the gap is visible in the text."""
    monkeypatch.setattr(settings, "vision_batch_pages", 1)
    monkeypatch.setattr(settings, "vision_max_failed_fraction", 0.34)
    stub_pdf(monkeypatch, 3)
    client = FakeVisionClient(
        response(f"{SEP}\none"), *[transient()] * 4, response(f"{SEP}\nthree")
    )
    monkeypatch.setattr(vision, "build_client", lambda: client)
    reported: list[tuple[int, int]] = []

    text = vision.pdf_to_markdown(
        "document.pdf", on_page_failed=lambda number, count: reported.append((number, count))
    )

    assert text == "one\n\n[page 2 could not be transcribed]\n\nthree"
    assert reported == [(2, 1)]
    assert client.closed


def test_too_many_failed_pages_fail_the_whole_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead key must not index 200 markers and call the upload a success."""
    monkeypatch.setattr(settings, "vision_batch_pages", 1)
    monkeypatch.setattr(settings, "vision_max_failed_fraction", 0.1)
    stub_pdf(monkeypatch, 3)
    client = FakeVisionClient(*[transient()] * 4)
    monkeypatch.setattr(vision, "build_client", lambda: client)

    with pytest.raises(ValueError, match="1 of 3 pages"):
        vision.pdf_to_markdown("document.pdf")
    assert client.closed


def test_each_document_gets_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio.run() closes its event loop, so a client reused across documents is dead.

    The symptom is APIConnectionError on the second upload and nothing at all on the first,
    which is why this asserts on construction count rather than on a successful call.
    """
    stub_pdf(monkeypatch, 1)
    built: list[FakeVisionClient] = []

    def factory() -> FakeVisionClient:
        built.append(FakeVisionClient(response(f"{SEP}\npage")))
        return built[-1]

    monkeypatch.setattr(vision, "build_client", factory)

    assert vision.pdf_to_markdown("first.pdf") == "page"
    assert vision.pdf_to_markdown("second.pdf") == "page"
    assert len(built) == 2
    assert [client.closed for client in built] == [True, True]


def test_pdf_to_markdown_rejects_an_empty_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision, "page_count", lambda path: 0)

    with pytest.raises(ValueError, match="No text extracted"):
        vision.pdf_to_markdown("document.pdf")


def test_pdf_to_markdown_rejects_an_oversized_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count is checked before any page is rendered, so an absurd PDF costs nothing."""
    monkeypatch.setattr(settings, "vision_max_pages", 5)
    monkeypatch.setattr(vision, "page_count", lambda path: 6)
    monkeypatch.setattr(
        vision, "build_client", lambda: pytest.fail("client built for a rejected PDF")
    )

    with pytest.raises(ValueError, match="limit is 5"):
        vision.pdf_to_markdown("document.pdf")


def gone() -> NotFoundError:
    request = httpx2.Request("POST", "https://vision.test")
    response = httpx2.Response(404, request=request)
    return NotFoundError("model not found", response=response, body=None)


@pytest.mark.asyncio
async def test_a_retired_model_id_goes_straight_to_the_fallback(
    no_sleeping: list[float],
) -> None:
    """404 is not worth a second try on the same model, and it is what the fallback is for."""
    client = FakeVisionClient(gone(), response(f"{SEP}\none"))

    assert await vision.pages_to_markdown(client, [b"a"]) == ["one"]
    assert client.models == ["primary", "fallback"]
    assert no_sleeping == []


@pytest.mark.asyncio
async def test_both_model_ids_gone_names_the_models() -> None:
    client = FakeVisionClient(gone(), gone())

    with pytest.raises(ValueError, match="primary, fallback"):
        await vision.pages_to_markdown(client, [b"a"])


@pytest.mark.asyncio
async def test_a_busy_primary_and_a_missing_fallback_stay_a_page_failure(
    no_sleeping: list[float],
) -> None:
    """The 404 must not upgrade a busy page into "no vision model configured".

    That message raises ValueError, which is outside PAGE_FAILURES, so the document would fail
    whole instead of losing the one page the ceiling allows.
    """
    client = FakeVisionClient(transient(), transient(), gone())

    with pytest.raises(APIConnectionError):
        await vision.pages_to_markdown(client, [b"a"])
    assert client.models == ["primary", "primary", "fallback"]
