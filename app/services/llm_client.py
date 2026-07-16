from google import genai
from google.genai import types

from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

async def generate_response(prompt: str):
    responses = await client.aio.models.generate_content_stream(
        model=settings.main_model,
        contents=prompt,
        config={
            'temperature': 1.0,
            'system_instruction': settings.rag_prompt,
            'max_output_tokens': 6000,
            'thinking_config': {
                'thinking_level': 'high',
                'include_thoughts': False
            },
        }
    )
    async for chunk in responses:
        if chunk.text:
            yield chunk.text


async def generate_title(query: str) -> str:
    response = await client.aio.models.generate_content(
        model=settings.summary_model,
        contents=query,
        config={
            'temperature': 0.3,
            'automatic_function_calling': {'disable': True},
            'system_instruction': (
                "Generate a short chat title from the user's first message: "
                "3-5 words, same language as the message, "
                "no quotes and no trailing punctuation."
            ),
            'max_output_tokens': 64,
            'thinking_config': {'thinking_budget': 0},
        }
    )
    return (response.text or "").strip().strip('"')


async def generate_summary(content: str) -> str:
    response = await client.aio.models.generate_content(
        model=settings.summary_model,
        contents=content,
        config={
            'temperature': 1.0,
            'automatic_function_calling': {'disable': True},
            'system_instruction': settings.summary_prompt,
            'max_output_tokens': 16000,
            'thinking_config': {
                'thinking_level': 'high',
                'include_thoughts': False
            }
        }
    )
    return response.text