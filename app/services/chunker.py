import re

from app.config import settings

# No whitespace inside, so the word-level split can never cut a marker in two.
PAGE_MARKER = re.compile(r'<page-(\d+)/>')


def page_marker(number: int) -> str:
    return f'<page-{number}/>'


def split_page_markers(chunks: list[str]) -> list[tuple[str, list[int]]]:
    """Each chunk without its markers, and the pages its text comes from.

    Markers split a chunk into segments, and a page counts only if its segment has text. A
    segment after a marker is that marker's page. The text before a chunk's first marker
    belongs to the page preceding that marker in the document, which is not always the last
    marker of the previous chunk: the overlap can repeat text from before it. A chunk with no
    marker is on the last page seen, so the chunks have to be walked in order. Text before
    any marker maps to no page.
    """
    result: list[tuple[str, list[int]]] = []
    order: list[int] = []
    for chunk in chunks:
        parts = PAGE_MARKER.split(chunk)
        segments, numbers = parts[0::2], [int(n) for n in parts[1::2]]
        order += [n for n in dict.fromkeys(numbers) if n not in order]
        if numbers:
            at = order.index(numbers[0])
            leading = order[at - 1] if at > 0 else None
        else:
            leading = order[-1] if order else None
        owners = [leading, *numbers]
        pages = list(dict.fromkeys(
            owner for segment, owner in zip(segments, owners)
            if owner is not None and segment.strip()
        ))
        text = re.sub(r'\n{3,}', '\n\n', PAGE_MARKER.sub('', chunk)).strip()
        if text:
            result.append((text, pages))
    return result


def chunk_with_pages(text: str, chunk_size: int, overlap: int) -> list[tuple[str, list[int]]]:
    return split_page_markers(smart_chunk_text(text, chunk_size, overlap))


def smart_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = text.split("\n\n")
    result = []
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) + len(buffer) + 2 <= chunk_size:
            buffer += "\n\n" + paragraph

        else:
            lines = paragraph.split("\n")
            for line in lines:
                if len(buffer) + len(line) + 1 <= chunk_size:
                    buffer += "\n" + line
                    continue

                if len(buffer) + len(line) + 1 > chunk_size:
                    words = line.split()
                    for word in words:
                        if len(buffer) + len(word) + 1 <= chunk_size:
                            buffer += " " + word
                            continue

                        if len(buffer) + len(word) + 1 > chunk_size:
                            result.append(buffer.strip())
                            tail = buffer[-overlap:] if len(buffer) > overlap else buffer
                            tail =  " ".join(tail.split()[1:])
                            buffer =  tail + " " + word

    if buffer:
        result.append(buffer)

    return result