import json
import re
import urllib.request
from datetime import datetime, timezone

from app.config import settings
from app.services import vision
from app.services.chunker import smart_chunk_text
from evals.corpus import CHUNKS, MD_DIR, PDF_DIR, SNAPSHOT, load_manifest, sha256

# Kept rather than retried: production indexes the same marker, and that is what is measured.
FAILED_PAGE = re.compile(r"\[page (\d+) could not be transcribed\]")


def fetch_pdf(doc: dict) -> bytes:
    path = PDF_DIR / f"{doc['id']}.pdf"
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(f"https://arxiv.org/pdf/{doc['arxiv']}") as response:
            data = response.read()
    if sha256(data) != doc["pdf_sha256"]:
        raise SystemExit(f"{doc['id']}: PDF sha256 differs from corpus.toml")
    if not path.exists():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def transcript_key(doc: dict) -> dict:
    return {
        "pdf_sha256": doc["pdf_sha256"],
        "vision_model": settings.vision_model,
        "vision_fallback_model": settings.vision_fallback_model,
        "prompt_sha256": sha256(settings.vision_prompt.encode("utf-8")),
    }


def transcribe(doc: dict) -> str:
    path = MD_DIR / f"{doc['id']}.md"
    key_path = path.with_suffix(".json")
    key = transcript_key(doc)
    if path.exists() and key_path.exists() and json.loads(key_path.read_text()) == key:
        return path.read_text(encoding="utf-8")
    print(f"{doc['id']}: transcribing with {settings.vision_model}")
    text = vision.pdf_to_markdown(str(PDF_DIR / f"{doc['id']}.pdf"))
    MD_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2))
    return text


def main() -> None:
    docs = load_manifest()
    snapshot = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": settings.chunk_size,
        "overlap": settings.overlap,
        "docs": {},
    }
    rows = []
    for doc in docs:
        fetch_pdf(doc)
        text = transcribe(doc)
        chunks = smart_chunk_text(text, settings.chunk_size, settings.overlap)
        rows += [{"doc": doc["id"], "i": i, "text": c} for i, c in enumerate(chunks)]
        snapshot["docs"][doc["id"]] = {
            **transcript_key(doc),
            "md_sha256": sha256(text.encode("utf-8")),
            "chunks": len(chunks),
            "failed_pages": [int(n) for n in FAILED_PAGE.findall(text)],
        }
        print(f"{doc['id']}: {len(text)} chars, {len(chunks)} chunks")
    with open(CHUNKS, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    snapshot["chunks_sha256"] = sha256(CHUNKS.read_bytes())
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
