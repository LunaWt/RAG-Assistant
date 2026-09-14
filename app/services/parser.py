import docx
import pandas as pd
from pptx import Presentation
from pathlib import Path

from app.services import vision

SUPPORTED_SUFFIXES = frozenset({'.pdf', '.md', '.txt', '.docx', '.xlsx', '.pptx'})


def _read_text_file(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            with open(file_path, encoding=encoding) as file:
                text = file.read()
                if text.strip():
                    return text
        except UnicodeDecodeError:
            continue
    raise ValueError("No text extracted")


def extract_text(file_path: str, on_progress=None) -> str:

    suffix = Path(file_path).suffix

    # Every PDF goes through vision, not only the ones pdfplumber cannot read. Measured
    # 2026-09-14 on page 10 of a 49-page arXiv paper: pdfplumber returned non-empty text for
    # that page, so a fallback-on-empty trigger would never have fired, and the text it
    # returned had the table rows flattened and the rotated axis labels reversed (") ( ssoL").
    # Extraction succeeding is not extraction being right.
    if suffix == '.pdf':
        return vision.pdf_to_markdown(str(file_path), on_progress)

    elif suffix in ('.md', '.txt'):
        return _read_text_file(file_path)

    elif suffix == '.docx':
        doc = docx.Document(file_path)
        paragraphs = [text for p in doc.paragraphs if (text := p.text.strip())]
        if paragraphs:
            return '\n'.join(paragraphs)
        else:
            raise ValueError('No text extracted')
    
    elif suffix == '.xlsx':
        df_dict = pd.read_excel(file_path, sheet_name=None)
        text = [f'{sheet_name}: {df.to_string(index=False).strip()}' 
                for sheet_name, df in df_dict.items() 
                if not df.empty
        ]
        if text:
            return '\n\n'.join(text)
        else:
            raise ValueError('No text extracted')

    elif suffix == '.pptx':
        presentation = Presentation(file_path)
        slides = [
            text
            for slide in presentation.slides
            for shape in slide.shapes 
            if hasattr(shape, 'text') and (text := shape.text)
            ]
        if slides:
            return '\n'.join(slides)
        else:
            raise ValueError('No text extracted')
    
    else:
        raise ValueError('Unsupported file type')
    