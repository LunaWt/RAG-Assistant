from app.config import settings

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