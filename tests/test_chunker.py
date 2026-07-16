from app.services.chunker import smart_chunk_text


def test_smart_chunk_text_returns_list():
    text = "Paragraph one.\n\nParagraph two with more words."
    chunks = smart_chunk_text(text, chunk_size=50, overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)

def test_smart_chunk_text_success_line():
    text = " aaaa bbbb"
    chunks = smart_chunk_text(text, chunk_size=10, overlap=2)
    assert ' aaaa bbbb' in chunks

def test_smart_chunk_text_success_word():
    text = "aaaa bbbb cccc"
    chunks = smart_chunk_text(text, chunk_size=9, overlap=0)
    assert 'bbbb' in ' '.join(chunks)
