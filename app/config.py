from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf_8')

    ## LLM
    gemini_api_key: str
    hf_token: str
    main_model: str = 'gemma-4-26b-a4b-it'
    summary_model: str = 'gemini-3.1-flash-lite'
    
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
    
    ## RAG
    chunk_size: int = 1000
    overlap: int = 50
    embedding_model: str = "BAAI/bge-m3"
    dimension: int = 1024

    ## STORAGE
    storage_dir: str = 'app/storage'


settings = Settings()
