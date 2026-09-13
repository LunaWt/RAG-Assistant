import base64
import io

import pypdfium2

from app.config import settings
from app.services.llm_client import client, first_text


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


def page_count(pdf_path: str) -> int:
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


async def page_to_markdown(png: bytes, model: str | None = None) -> str:
    data_url = 'data:image/png;base64,' + base64.b64encode(png).decode()
    response = await client.chat.completions.create(
        model=model or settings.vision_model,
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': settings.vision_prompt},
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=settings.vision_max_tokens,
    )
    # Truncation is the silent failure here: a page cut off at max_tokens returns valid-looking
    # Markdown that is simply missing its tail, and nothing downstream can tell.
    if response.choices and response.choices[0].finish_reason == 'length':
        raise ValueError('Vision model hit the output limit; page transcript is incomplete')
    text = first_text(response).strip()
    if not text:
        raise ValueError('Vision model returned no text')
    return text
