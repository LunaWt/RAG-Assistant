from types import SimpleNamespace

import pytest

import app.services.vision as vision
from app.config import settings

SEP = settings.vision_page_separator


def response(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class FakeVisionClient:
    """Replays one prepared reply per request and records how many images each carried."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self._replies = list(replies)
        self.batch_sizes: list[int] = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        self.batch_sizes.append(sum(1 for p in content if p["type"] == "image_url"))
        if not self._replies:
            raise AssertionError("unscripted vision request")
        return self._replies.pop(0)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_batch_splits_into_one_entry_per_page() -> None:
    client = FakeVisionClient(response(f"{SEP}\none\n{SEP}\ntwo"))

    assert await vision.pages_to_markdown(client, [b"a", b"b"]) == ["one", "two"]
    assert client.batch_sizes == [2]


@pytest.mark.asyncio
async def test_truncated_single_page_raises() -> None:
    """One page that will not fit the output budget is a real failure, not a retry case."""
    client = FakeVisionClient(response(f"{SEP}\none", finish_reason="length"))

    with pytest.raises(ValueError, match="output limit"):
        await vision.pages_to_markdown(client, [b"a"])


@pytest.mark.asyncio
async def test_truncated_batch_falls_back_to_one_call_per_page() -> None:
    """A batch overflowing the output cap is the case splitting it actually fixes."""
    client = FakeVisionClient(
        response(f"{SEP}\ncut off here", finish_reason="length"),
        response(f"{SEP}\none"),
        response(f"{SEP}\ntwo"),
    )

    assert await vision.pages_to_markdown(client, [b"a", b"b"]) == ["one", "two"]
    assert client.batch_sizes == [2, 1, 1]


@pytest.mark.asyncio
async def test_short_batch_falls_back_to_one_call_per_page() -> None:
    """Two pages in, one page back: the batch is retried page by page rather than accepted."""
    client = FakeVisionClient(
        response(f"{SEP}\nonly one"),
        response(f"{SEP}\none"),
        response(f"{SEP}\ntwo"),
    )

    assert await vision.pages_to_markdown(client, [b"a", b"b"]) == ["one", "two"]
    assert client.batch_sizes == [2, 1, 1]


@pytest.mark.asyncio
async def test_single_page_mismatch_is_not_retried_forever() -> None:
    client = FakeVisionClient(response(f"{SEP}\na\n{SEP}\nb"))

    with pytest.raises(vision.PageCountMismatch):
        await vision.pages_to_markdown(client, [b"a"])
    assert client.batch_sizes == [1]


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
