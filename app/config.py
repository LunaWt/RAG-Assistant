from urllib.parse import urlparse

from pydantic import PositiveInt, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf_8')

    ## LLM
    llm_api_key: str
    llm_base_url: str = 'https://openrouter.ai/api/v1'
    llm_proxy_url: str | None = None
    llm_timeout: float = 120.0
    llm_connect_timeout: float = 5.0
    hf_token: str
    main_model: str = 'openrouter/free'
    summary_model: str = 'inclusionai/ling-3.0-flash-fin:free'
    
    ## PROMPTS
    main_agent_prompt: str = """
    You are a helpful assistant with three tools:
    - search_knowledge_base: the user's uploaded documents (RAG)
    - web_search: current or external info not in your training data
    - calculator: precise arithmetic

    Answer directly from your own knowledge whenever you can. Call a tool ONLY when the
    question genuinely needs it: the user's documents, fresh/real-time facts, or exact math.
    Prefer ONE well-formed call per need. Never repeat the same search with reworded
    queries — if results are weak, reason over what you already have and answer with caveats.
    As soon as you have enough to answer, stop calling tools and respond. Be fast: aim for
    0-3 tool calls total, more only if the task truly requires it.
    Questions like "what is in your context / knowledge base" can be answered from the
    document list below without any tool call.
    """
    rag_prompt: str = """
        Answer only with provided information from chunks
        (now I am just testing it and playing around so answer from your knowledge)"""
    summary_prompt: str = """Summarize the provided web-scraped text. 
        Focus on extracting key logic and all technical/numerical data. 
        Keep it concise but ensure no critical facts or numbers are omitted. 
        Format: coherent paragraph(s)."""
    
    ## VISION
    # Vision runs on Google AI Studio, not on the chat provider: the OpenRouter free pool
    # returned 429 on three of four vision models tested 2026-09-13, which a page loop cannot
    # absorb. Separate key and base URL, so the chat provider stays swappable on its own.
    vision_api_key: str = ''
    vision_base_url: str = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    vision_proxy_url: str | None = None
    vision_timeout: float = 300.0
    # Measured 2026-09-14 through the tunnel, page 10 of a 49-page arXiv PDF with three tables:
    # gemini-3.5-flash-lite returned every cell in 5.8 s. gemma-4-31b-it is not a fallback
    # candidate and re-testing it needs a reason. It returns HTTP 500 INTERNAL for a plain
    # text-only prompt with no image at all, and on the native generateContent endpoint too, so
    # the failure is upstream of anything configurable here; gemma-4-26b-a4b-it behaves the
    # same. Its 16 000 TPM free-tier budget also rules it out on throughput: one 10-page batch
    # is ~24 000 tokens, more than a whole minute's allowance, capping it near 6.6 pages/min
    # against 102 for flash-lite.
    vision_model: str = 'gemini-3.5-flash-lite'
    # 150 dpi is the usual OCR floor; below it small type breaks, above it the base64 payload
    # grows ~4/3 of an already quadratic pixel count for no accuracy gain.
    vision_dpi: int = 150
    # The output cap is what bounds a batch, not the input context. Same measurement:
    # ~1100 input tokens and ~1325 output tokens per page, against 1M input and 65 536 output.
    # Input allows ~900 pages; output allows ~49. Ten pages is ~13 k output tokens, 5x headroom,
    # and it keeps one failed batch from costing the whole document.
    vision_batch_pages: PositiveInt = 10
    # A ceiling so nobody feeds in a 10 000-page scan and burns the daily request quota on one
    # upload. Luna's number, 14 Sep 2026: high enough that no real document hits it.
    vision_max_pages: PositiveInt = 2000
    vision_max_tokens: int = 32000
    vision_page_separator: str = '---PAGE---'
    # {separator} is substituted with vision_page_separator before the call. Hardcoding the
    # marker here instead would let an override of one setting desynchronise it from the
    # splitter, and the symptom is a silent loss of batching rather than an error.
    vision_prompt: str = """Transcribe every page image into Markdown, in order.
        Begin each page with a line containing only {separator} including the first.
        Reproduce the text exactly, in reading order. Use Markdown headings for headings and
        Markdown tables for tables, preserving every cell. Do not summarise, explain, translate
        or add commentary. Output the page content only."""

    ## RAG
    chunk_size: int = 3000
    overlap: int = 300
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    embedding_batch_size: int = 16
    dimension: int = 1024

    ## STORAGE
    storage_dir: str = 'app/storage'

    @field_validator('llm_base_url', 'vision_base_url')
    @classmethod
    def require_encrypted_transport(cls, url: str, info: ValidationInfo) -> str:
        # The API key travels in the Authorization header of every request, so a plain http
        # URL sends the key in the clear. A redirect to https does not save it: the header
        # is already on the wire. Loopback stays allowed for a local Ollama or vLLM.
        parsed = urlparse(url)
        if parsed.scheme != 'https' and parsed.hostname not in LOOPBACK_HOSTS:
            name = (info.field_name or '').upper()
            raise ValueError(f'{name} must use https (got {parsed.scheme or url!r})')
        return url


settings = Settings()
