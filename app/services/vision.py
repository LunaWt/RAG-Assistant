import asyncio
import base64
import io
import logging

import httpx2
import pypdfium2
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from app.config import settings

logger = logging.getLogger(__name__)


class PageCountMismatch(RuntimeError):
    """The model returned a different number of pages than the batch contained."""


class BatchTruncated(ValueError):
    """The model hit its output cap before finishing the batch.

    A ValueError because a truncated *single* page is a genuine parse failure that the upload
    path already knows how to report. A truncated multi-page batch is not: it means the batch
    asked for more Markdown than the output budget allows, and splitting it fixes that.
    """


# Retried, because they clear on their own: the free tier answers 429 on burst and Google
# returns 500 INTERNAL under load.
TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
# What "this page did not come back" means for the caller that has to choose between a marker
# and a failed upload. The two batch errors join the list only because on a *single* page they
# are no longer splittable: one page of Markdown that will not fit the output cap, or a model
# answering with a page count it was not sent.
PAGE_FAILURES = (*TRANSIENT_ERRORS, PageCountMismatch, BatchTruncated)
# Not retried on the same model and not a page failure either: a model id that no longer
# resolves is the case the fallback exists for. Google retires ids (gemini-3.1-flash-lite-preview
# went on 2026-05-25) and a config can outlive one.
MODEL_GONE = (NotFoundError,)


def build_client() -> AsyncOpenAI:
    """A client for one document.

    Deliberately not a module-level singleton. Parsing runs in a worker thread and drives the
    async calls with asyncio.run(), which creates and then closes a fresh event loop per
    document. An httpx connection pool built under the first loop is unusable under the
    second, and the failure surfaces as APIConnectionError on the *second* upload, not the
    first — measured 2026-09-14, call 1 fine, call 2 dead.
    """
    timeout = httpx2.Timeout(settings.vision_timeout, connect=settings.llm_connect_timeout)
    # Only reuse the chat key when both point at the same provider. Falling back
    # unconditionally would post the OpenRouter key to Google on any machine that set up chat
    # but not vision, and a leaked credential does not announce itself.
    api_key = settings.vision_api_key
    if not api_key and settings.vision_base_url == settings.llm_base_url:
        api_key = settings.llm_api_key
    if not api_key:
        raise ValueError(
            'VISION_API_KEY is required to parse PDFs when VISION_BASE_URL differs from '
            'LLM_BASE_URL'
        )
    kwargs = {
        'api_key': api_key,
        'base_url': settings.vision_base_url,
        'timeout': timeout,
        'max_retries': 0,
    }
    proxy = settings.vision_proxy_url or settings.llm_proxy_url
    if proxy:
        kwargs['http_client'] = httpx2.AsyncClient(proxy=proxy, timeout=timeout)
    return AsyncOpenAI(**kwargs)


def page_count(pdf_path: str) -> int:
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_page(pdf_path: str, page_number: int, dpi: int | None = None) -> bytes:
    scale = (dpi or settings.vision_dpi) / 72
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        image = pdf[page_number].render(scale=scale).to_pil()
    finally:
        pdf.close()
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def _image_part(png: bytes) -> dict:
    return {
        'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()},
    }


def _split_pages(text: str) -> list[str]:
    parts = text.split(settings.vision_page_separator)
    if parts and not parts[0].strip():
        parts = parts[1:]
    return [part.strip() for part in parts]


async def _transcribe(client: AsyncOpenAI, pngs: list[bytes], model: str) -> list[str]:
    prompt = settings.vision_prompt.replace(
        '{separator}', settings.vision_page_separator
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    *(_image_part(png) for png in pngs),
                ],
            }
        ],
        temperature=0.0,
        max_tokens=settings.vision_max_tokens,
    )
    if not response.choices:
        raise ValueError('Vision model returned no choices')
    choice = response.choices[0]
    # Truncation is the silent failure here: a batch cut off at max_tokens returns valid-looking
    # Markdown that is simply missing its tail, and nothing downstream can tell.
    if choice.finish_reason == 'length':
        raise BatchTruncated(
            'Vision model hit the output limit; page transcript is incomplete'
        )
    pages = _split_pages(choice.message.content or '')
    # The other silent failure: the model transcribes six of ten pages, emits six separators and
    # still finishes with "stop". Without this check the document loses four pages and no error
    # ever reaches the job store.
    if len(pages) != len(pngs):
        raise PageCountMismatch(f'sent {len(pngs)} pages, got {len(pages)} back')
    return pages


def _retry_delay(error: Exception, attempt: int) -> float:
    """Exponential backoff, overridden by the server's own Retry-After when it sends one."""
    headers = getattr(getattr(error, 'response', None), 'headers', None)
    requested = None
    if headers is not None:
        try:
            requested = float(headers.get('retry-after') or '')
        except (AttributeError, TypeError, ValueError):
            requested = None
    if requested is None:
        requested = settings.vision_retry_backoff * 2**attempt
    return min(requested, settings.vision_retry_max_sleep)


def _models() -> list[str]:
    fallback = settings.vision_fallback_model
    if fallback and fallback != settings.vision_model:
        return [settings.vision_model, fallback]
    return [settings.vision_model]


async def pages_to_markdown(client: AsyncOpenAI, pngs: list[bytes]) -> list[str]:
    """One batch of page images, retried on the primary model and then on the fallback.

    Only transient errors are retried. A truncated batch or a page-count mismatch is not
    transient — the same images and the same prompt produce it again — so it leaves here
    immediately for the caller to split into single pages.
    """
    transient: Exception | None = None
    gone: Exception = RuntimeError('no vision model configured')
    for model in _models():
        for attempt in range(settings.vision_attempts):
            try:
                return await _transcribe(client, pngs, model)
            except MODEL_GONE as error:
                gone = error
                logger.warning('Vision model %s does not resolve: %s', model, error)
                break
            except TRANSIENT_ERRORS as error:
                transient = error
                logger.warning(
                    'Vision request failed (%s, attempt %d/%d, %d pages): %s',
                    model, attempt + 1, settings.vision_attempts, len(pngs), error,
                )
                if attempt + 1 < settings.vision_attempts:
                    await asyncio.sleep(_retry_delay(error, attempt))
    if transient is not None:
        # One model missing and the other merely busy is still a page problem, so the transient
        # error wins: it is in PAGE_FAILURES, so the caller can isolate the page and carry on.
        raise transient
    # Every model 404s. Not a marker in the text: every page will fail the same way, and the
    # operator needs to read the model name rather than a count of unreadable pages. ValueError,
    # so index_document reports this sentence instead of a generic server error.
    raise ValueError(
        f'No configured vision model is available ({", ".join(_models())}): {gone}'
    ) from gone


def _placeholder(number: int) -> str:
    # The marker is indexed with the rest of the document on purpose: a gap that retrieval can
    # surface beats a document that quietly lost a page. Fixed wording, so it cannot be
    # mistaken for the document's own text.
    return f'[page {number} could not be transcribed]'


async def _transcribe_batch(
    client: AsyncOpenAI, pngs: list[bytes], first_number: int, note_failed
) -> list[str]:
    """A batch, degraded to one call per page as soon as the batch as a whole fails.

    Splitting fixes three different failures: a batch whose Markdown overflows the output cap,
    a model that answers with fewer pages than it was sent, and one unreadable page taking nine
    readable ones down with it. What survives the split becomes a marker.
    """
    if len(pngs) > 1:
        try:
            return await pages_to_markdown(client, pngs)
        except PAGE_FAILURES:
            logger.warning(
                'Vision batch of %d pages starting at %d failed; retrying page by page',
                len(pngs), first_number,
            )
    transcripts: list[str] = []
    for offset, png in enumerate(pngs):
        number = first_number + offset
        try:
            transcripts.extend(await pages_to_markdown(client, [png]))
        except PAGE_FAILURES:
            logger.exception('Vision could not transcribe page %d', number)
            transcripts.append(_placeholder(number))
            note_failed(number)
    return transcripts


def pdf_to_markdown(pdf_path: str, on_progress=None, on_page_failed=None) -> str:
    total = page_count(pdf_path)
    if not total:
        raise ValueError('No text extracted')
    # ValueError on purpose: index_document turns it into a job error carrying this message,
    # so the user is told the page count rather than seeing a generic server failure.
    if total > settings.vision_max_pages:
        raise ValueError(
            f'PDF has {total} pages; the limit is {settings.vision_max_pages}'
        )
    if on_progress:
        on_progress(0, total)

    async def run() -> list[str]:
        client = build_client()
        failed: list[int] = []

        def note_failed(number: int) -> None:
            failed.append(number)
            if on_page_failed:
                on_page_failed(number, len(failed))
            # The ceiling is what separates a scanned page the model choked on from a spent
            # quota. Without it a key that died at page 3 of 200 indexes 197 markers and calls
            # the upload a success.
            if len(failed) > total * settings.vision_max_failed_fraction:
                raise ValueError(
                    f'Vision could not transcribe {len(failed)} of {total} pages '
                    f'({", ".join(str(n) for n in failed)}); the document was not indexed'
                )

        try:
            transcripts: list[str] = []
            for start in range(0, total, settings.vision_batch_pages):
                batch = range(start, min(start + settings.vision_batch_pages, total))
                pngs = [render_page(pdf_path, number) for number in batch]
                transcripts.extend(
                    await _transcribe_batch(client, pngs, start + 1, note_failed)
                )
                if on_progress:
                    on_progress(len(transcripts), total)
            return transcripts
        finally:
            await client.close()

    text = '\n\n'.join(page for page in asyncio.run(run()) if page)
    if not text.strip():
        raise ValueError('No text extracted')
    return text
