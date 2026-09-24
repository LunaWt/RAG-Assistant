import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus"
PDF_DIR = CORPUS / "pdf"
MD_DIR = CORPUS / "md"
CHUNKS = CORPUS / "chunks.jsonl"
SNAPSHOT = CORPUS / "snapshot.json"


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> list[dict]:
    with open(ROOT / "corpus.toml", "rb") as f:
        return tomllib.load(f)["doc"]


def load_chunks() -> list[dict]:
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_verified_docs() -> dict[str, str]:
    if not SNAPSHOT.exists():
        raise SystemExit("no corpus snapshot: run py -3 -m evals.build_corpus first")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if sha256(CHUNKS.read_bytes()) != snapshot["chunks_sha256"]:
        raise SystemExit("chunks.jsonl differs from the snapshot: rerun evals.build_corpus")
    docs = {}
    for doc_id, meta in snapshot["docs"].items():
        text = (MD_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
        if sha256(text.encode("utf-8")) != meta["md_sha256"]:
            raise SystemExit(f"{doc_id}.md differs from the snapshot: rerun evals.build_corpus")
        docs[doc_id] = text
    return docs
