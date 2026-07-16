from pathlib import Path

import pytest

import app.services.parser as parser_module
from app.services.parser import extract_text


def test_extract_text_wrong_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(file_path="dir/folder/file.csv")


def test_extract_text_success(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("Начали", encoding="utf-8")

    assert extract_text(str(file_path)) == "Начали"


def test_extract_text_reads_markdown(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Заголовок\n\nТекст заметки", encoding="utf-8")

    assert extract_text(str(file_path)) == "# Заголовок\n\nТекст заметки"


def test_extract_text_reads_cp1251_text(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_bytes("Текст в cp1251".encode("cp1251"))

    assert extract_text(str(file_path)) == "Текст в cp1251"


def test_extract_text_rejects_empty_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(ValueError, match="No text extracted"):
        extract_text(str(file_path))


def test_extract_text_reads_pdf_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakePdf:
        pages = [FakePage("First page"), FakePage(None), FakePage("Second page")]

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr(parser_module.pdfplumber, "open", lambda _: FakePdf())

    assert extract_text("document.pdf") == "First page\nSecond page"


def test_extract_text_rejects_pdf_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def extract_text(self) -> None:
            return None

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr(parser_module.pdfplumber, "open", lambda _: FakePdf())

    with pytest.raises(ValueError, match="No text extracted"):
        extract_text("document.pdf")


def test_extract_text_reads_docx_paragraphs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeParagraph:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeDocument:
        paragraphs = [
            FakeParagraph(" First paragraph "),
            FakeParagraph(""),
            FakeParagraph("Second paragraph"),
        ]

    monkeypatch.setattr(parser_module.docx, "Document", lambda _: FakeDocument())

    assert extract_text("document.docx") == "First paragraph\nSecond paragraph"


def test_extract_text_rejects_docx_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeParagraph:
        text = "  "

    class FakeDocument:
        paragraphs = [FakeParagraph()]

    monkeypatch.setattr(parser_module.docx, "Document", lambda _: FakeDocument())

    with pytest.raises(ValueError, match="No text extracted"):
        extract_text("document.docx")


def test_extract_text_reads_xlsx_sheets(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataFrame:
        empty = False

        def to_string(self, index: bool) -> str:
            assert index is False
            return " name  score \nAlice     10 "

    monkeypatch.setattr(
        parser_module.pd,
        "read_excel",
        lambda _, sheet_name: {"Scores": FakeDataFrame()},
    )

    assert extract_text("document.xlsx") == "Scores: name  score \nAlice     10"


def test_extract_text_rejects_xlsx_without_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataFrame:
        empty = True

    monkeypatch.setattr(
        parser_module.pd,
        "read_excel",
        lambda _, sheet_name: {"Empty sheet": FakeDataFrame()},
    )

    with pytest.raises(ValueError, match="No text extracted"):
        extract_text("document.xlsx")


def test_extract_text_reads_pptx_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    class TextShape:
        def __init__(self, text: str) -> None:
            self.text = text

    class ImageShape:
        pass

    class FakeSlide:
        shapes = [TextShape("Title"), ImageShape(), TextShape("Slide body")]

    class FakePresentation:
        slides = [FakeSlide()]

    monkeypatch.setattr(parser_module, "Presentation", lambda _: FakePresentation())

    assert extract_text("document.pptx") == "Title\nSlide body"


def test_extract_text_rejects_pptx_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TextShape:
        text = ""

    class FakeSlide:
        shapes = [TextShape()]

    class FakePresentation:
        slides = [FakeSlide()]

    monkeypatch.setattr(parser_module, "Presentation", lambda _: FakePresentation())

    with pytest.raises(ValueError, match="No text extracted"):
        extract_text("document.pptx")
