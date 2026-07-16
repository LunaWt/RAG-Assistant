from app.services.llm_client import client
from app.services.tools import search_knowledge_base, web_search, calculator
from app.services.vector_db import vector_db
from app.config import settings
from google.genai import types
import asyncio
import inspect

TOOLS = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
    "web_search": web_search,
}

async def run_tool(fc: types.FunctionCall) -> tuple[types.Part, list[dict] | None]:
    if fc.name not in TOOLS or not fc.args:
        return types.Part.from_function_response(
            name=fc.name,
            response={'result': "Tool not found or no arguments provided or arguments are empty"},
        ), []
    fn = TOOLS[fc.name]
    try:
        if inspect.iscoroutinefunction(fn):
            summary, searches_list = await fn(**fc.args)
            part = types.Part.from_function_response(
                name=fc.name,
                response={'result': summary if summary is not None else 'tool returned None'},
            )
            return part, searches_list
        else:
            summary = await asyncio.to_thread(fn, **fc.args)
            part = types.Part.from_function_response(
                name=fc.name,
                response={'result': summary if summary is not None else 'tool returned None'},
            )
            return part, []
    except Exception as e:
        return types.Part.from_function_response(
            name=fc.name,
            response={'result': f"Error running tool: {e}"},
        ), []
    


# def parse_turn(response) -> tuple[str, str, list]:
#     """Текст этого хода + tool calls (если есть)."""
#     partial_text = ""
#     thoughts = ""
#     for part in response.parts or []:
#         if part.text:
#             if part.thought:
#                 thoughts += part.thoughts
#             else:
#                 partial_text += part.text
          

#     calls = list(response.function_calls or [])
#     return thoughts, partial_text.strip(), calls


def _assistant_text(msg: dict) -> str:
    if msg.get("content"):
        return msg["content"]
    blocks = msg.get("blocks") or []
    parts = [b["text"] for b in blocks if b.get("type") == "answer" and b.get("text")]
    return "\n".join(parts)


def build_system_instruction() -> str:
    docs = vector_db.list_sources()
    base = settings.main_agent_prompt.strip()
    if not docs:
        return (
            f"{base}\n\n"
            "Knowledge base: empty. If the user asks about uploaded files, "
            "tell them to upload a document in the sidebar first."
        )
    listing = "\n".join(f"  - {name}" for name in docs)
    return (
        f"{base}\n\n"
        "Indexed documents in the knowledge base "
        "(use search_knowledge_base; pass filename to search within one file):\n"
        f"{listing}\n"
        "When the user asks to search RAG / the knowledge base, call search_knowledge_base "
        "with a concrete query derived from the conversation — do not ask what to search "
        "if the topic is already clear from chat history."
    )


def to_gemini_history(history: list[dict] | None) -> list[types.Content]:
    """Convert stored chat history into native Gemini turns.

    Each message becomes one Content. Gemini names the two roles 'user' and
    'model', so our 'assistant' maps to 'model'. Empty messages are skipped.
    """
    contents: list[types.Content] = []
    for msg in history or []:
        text = _assistant_text(msg)
        if not text.strip():
            continue
        role = "model" if msg.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


async def agent_loop(query: str, history: list[dict] | None = None):
    """
    Agent loop for the application.
    """
    max_iterations = 20
    iterations = 0
    retries = 10

    for attempt in range(retries):
        try:
            iterations = 0
            chat = client.aio.chats.create(
        model=settings.main_model, 
        config={
            'temperature': 1.0,
            'system_instruction': build_system_instruction(),
            'max_output_tokens': 16000,
            'thinking_config': {
                'thinking_level': 'high',
                'include_thoughts': True,
            },
            'tools': [search_knowledge_base, web_search, calculator],
            'automatic_function_calling': {'disable': True},
            },
        history=to_gemini_history(history),
        )
            calls = []
            async for chunk in await chat.send_message_stream(query):
                for part in chunk.parts or []:
                    if part.function_call:
                        calls.append(part.function_call)
                    if part.text:
                        if part.thought:
                            yield {"type": "thought_delta", "text": part.text}
                        else:
                            yield {"type": "text_delta", "text": part.text}
                    
            if not calls:
                yield {"type": "done"}
                return

            while iterations <= max_iterations:
                
                yield {"type": "tool_start", "name": [call.name for call in calls], "args": [call.args for call in calls]}
                results = await asyncio.gather(*[run_tool(fc) for fc in calls])
                parts = [p for p, _ in results]
                
                for fc, (_, hits) in zip(calls, results):
                    if hits:    
                        yield {"type": "tool_hits", "query": (fc.args or {}).get("query", ""), "hits": hits}
                calls = []

                async for chunk in await chat.send_message_stream(parts):
                    for part in chunk.parts or []:
                        if part.function_call:
                            calls.append(part.function_call)
                        if part.text:
                            if part.thought:
                                yield {"type": "thought_delta", "text": part.text}
                            else:
                                yield {"type": "text_delta", "text": part.text}
                if not calls:
                    break
                iterations += 1

            if calls:
                async for chunk in await chat.send_message_stream(
                    "Give final answer based on what you get, you're out of tool calls"
                ):
                    for part in chunk.parts or []:
                        if not part.text:
                            continue
                        if part.thought:
                            yield {"type": "thought_delta", "text": part.text}
                        else:
                            yield {"type": "text_delta", "text": part.text}

            yield {"type": "done"}
            return

        except Exception as e:
            print(f'Ошибка обработки {e}')
            yield {"type": "stream_reset"}
            if attempt == retries - 1:
                yield {"type": "error", "message": f"Ошибка агента после {retries} попыток: {e}"}
                return

async def main():
    async for chunk in agent_loop('Привет! Какая погода в калифорнии?'):
        print(chunk)

if __name__=="__main__":
    asyncio.run(main())
