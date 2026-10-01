import random
import re

from app.services.chunker import (
    chunk_with_pages,
    page_marker,
    smart_chunk_text,
    split_page_markers,
)


def test_a_chunk_without_a_marker_inherits_the_page_it_sits_on():
    chunks = [
        "<page-1/>\n\nintro",
        "middle of page one",
        "end of one\n\n<page-2/>\n\nstart of two",
        "<page-3/>\n\nthree",
    ]

    assert split_page_markers(chunks) == [
        ("intro", [1]),
        ("middle of page one", [1]),
        ("end of one\n\nstart of two", [1, 2]),
        ("three", [3]),
    ]


def test_overlap_keeps_the_page_its_repeated_text_came_from():
    """The overlap can carry text from before the previous chunk's last marker; and a
    marker with no text after it in a chunk adds no page to that chunk."""
    chunks = [
        "<page-5/>\n\naaa bbb <page-6/> ccc",
        "bbb <page-6/> ccc ddd <page-7/>",
        "<page-7/> eee",
    ]

    assert [pages for _, pages in split_page_markers(chunks)] == [[5, 6], [5, 6], [7]]


def test_every_chunk_lists_exactly_the_pages_its_words_come_from():
    """Through the real chunker, overlap included: each word names its page, so a chunk's
    pages must equal the set of pages named by its words, wherever the boundaries fall."""
    rng = random.Random(0)
    for _ in range(100):
        pages = []
        for page in range(1, rng.randint(2, 12) + 1):
            if rng.random() < 0.1:
                continue
            paragraphs = (
                " ".join(f"p{page}w{rng.randint(0, 999)}" for _ in range(rng.randint(5, 60)))
                for _ in range(rng.randint(1, 8))
            )
            pages.append(f"{page_marker(page)}\n\n" + "\n\n".join(paragraphs))

        for text, chunk_pages in chunk_with_pages("\n\n".join(pages), 3000, 300):
            named = {int(n) for n in re.findall(r"\bp(\d+)w", text)}
            assert set(chunk_pages) == named, text[:80]


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
