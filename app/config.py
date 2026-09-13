from urllib.parse import urlparse

from pydantic import field_validator
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
    # Measured 2026-09-13 on one table page: gemma-4-31b, gemma-4-26b-a4b and ling-3.0-flash-vl
    # all returned 429 "temporarily rate-limited upstream" from the shared free pool on the
    # first attempt; nex-n2.5-pro answered and reproduced every table cell. The free tier is
    # a shared pool, so the page loop needs a fallback model, not only a retry.
    vision_model: str = 'nex-agi/nex-n2.5-pro:free'
    # 150 dpi is the usual OCR floor; below it small type breaks, above it the base64 payload
    # grows ~4/3 of an already quadratic pixel count for no accuracy gain.
    vision_dpi: int = 150
    vision_max_tokens: int = 8000
    vision_prompt: str = """Transcribe this page into Markdown.
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

    @field_validator('llm_base_url')
    @classmethod
    def require_encrypted_transport(cls, url: str) -> str:
        # LLM_API_KEY travels in the Authorization header of every request, so a plain http
        # URL sends the key in the clear. A redirect to https does not save it: the header
        # is already on the wire. Loopback stays allowed for a local Ollama or vLLM.
        parsed = urlparse(url)
        if parsed.scheme != 'https' and parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError(f'LLM_BASE_URL must use https (got {parsed.scheme or url!r})')
        return url


settings = Settings()
