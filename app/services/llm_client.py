import httpx2
from openai import AsyncOpenAI

from app.config import settings

# A reasoning model spends this budget before emitting any content, so a tight cap returns
# finish_reason="length" and an empty title. Measured 2026-09-11 on openai/gpt-oss-20b:
# 64 tokens gave '', 512 gave the title.
TITLE_MAX_TOKENS = 512

TITLE_PROMPT = (
    "Generate a short chat title from the user's first message: "
    "3-5 words, same language as the message, "
    "no quotes and no trailing punctuation."
)


def build_client() -> AsyncOpenAI:
    # A dropped SYN costs the OS SYN-retry budget, ~21 s on Windows, before the socket gives
    # up; measured 2026-09-11, 4 of 8 connections to openrouter.ai from this machine die that
    # way. A short connect deadline hands the failure to agent_loop's retry, while the read
    # deadline stays long enough for a slow model.
    timeout = httpx2.Timeout(settings.llm_timeout, connect=settings.llm_connect_timeout)
    kwargs = {
        'api_key': settings.llm_api_key,
        'base_url': settings.llm_base_url,
        'timeout': timeout,
        # agent_loop owns retries; the SDK's own default of 2 would multiply them.
        'max_retries': 0,
    }
    if settings.llm_proxy_url:
        kwargs['http_client'] = httpx2.AsyncClient(
            proxy=settings.llm_proxy_url, timeout=timeout
        )
    return AsyncOpenAI(**kwargs)


client = build_client()


def first_text(response) -> str:
    """Text of the first choice, or '' — a refusal or a truncated turn leaves it empty."""
    if not response.choices:
        return ''
    return response.choices[0].message.content or ''


async def generate_title(query: str) -> str:
    response = await client.chat.completions.create(
        model=settings.summary_model,
        messages=[
            {'role': 'system', 'content': TITLE_PROMPT},
            {'role': 'user', 'content': query},
        ],
        temperature=0.3,
        max_tokens=TITLE_MAX_TOKENS,
    )
    return first_text(response).strip().strip('"')


async def generate_summary(content: str) -> str:
    response = await client.chat.completions.create(
        model=settings.summary_model,
        messages=[
            {'role': 'system', 'content': settings.summary_prompt},
            {'role': 'user', 'content': content},
        ],
        temperature=1.0,
        max_tokens=16000,
    )
    return first_text(response)
