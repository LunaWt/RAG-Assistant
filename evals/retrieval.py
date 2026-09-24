import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings
from evals.check_labels import check, load_questions
from evals.corpus import ROOT, SNAPSHOT, load_chunks, load_verified_docs, normalize, sha256

KS = (1, 3, 5, 10)
REPORT = ROOT / "results" / "retrieval.json"


def fact_ranks(ranked: list[dict], evidence: list[dict]) -> list[int | None]:
    ranks = []
    for fact in evidence:
        quotes = [normalize(q) for q in fact["quotes"]]
        hits = (
            rank
            for rank, chunk in enumerate(ranked, 1)
            if chunk["doc"] == fact["doc"] and any(q in normalize(chunk["text"]) for q in quotes)
        )
        ranks.append(next(hits, None))
    return ranks


def score(ranks: list[int | None]) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    recall = {f"recall@{k}": sum(r <= k for r in found) / len(ranks) for k in KS}
    return recall | {"rr": 1 / min(found) if found else 0.0}


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    for split in ("dev", "test"):
        scored = [r["scores"] for r in rows if r["split"] == split and r["scores"]]
        means = {m: round(sum(s[m] for s in scored) / len(scored), 4) for m in scored[0]}
        means["mrr"] = means.pop("rr")
        summary[split] = {"answerable": len(scored)} | means
    return summary


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance(model: SentenceTransformer, questions_bytes: bytes) -> dict:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "../app", ".", ":!results")),
        "questions_sha256": sha256(questions_bytes),
        "snapshot_sha256": sha256(SNAPSHOT.read_bytes()),
        "chunks_sha256": snapshot["chunks_sha256"],
        "chunk_size": snapshot["chunk_size"],
        "overlap": snapshot["overlap"],
        "embedding_model": settings.embedding_model,
        "embedding_revision": model.model_card_data.base_model_revision,
        "device": str(model.device),
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
    }


def rank(model: SentenceTransformer, questions: list[dict], chunks: list[dict]) -> list[list[dict]]:
    chunk_vecs = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, batch_size=settings.embedding_batch_size
    )
    query_vecs = model.encode([q["text"] for q in questions], normalize_embeddings=True)
    similarity = query_vecs @ chunk_vecs.T
    top = np.argsort(-similarity, axis=1, kind="stable")[:, : max(KS)]
    return [[chunks[i] for i in row] for row in top]


def question_row(question: dict, ranked: list[dict]) -> dict:
    ranks = fact_ranks(ranked, question.get("evidence", []))
    return {
        "id": question["id"],
        "split": question["split"],
        "fact_ranks": ranks,
        "scores": score(ranks) if ranks else None,
        "top": [f"{c['doc']}:{c['i']}" for c in ranked],
    }


def main() -> None:
    docs = load_verified_docs()
    chunks = load_chunks()
    questions = load_questions()
    if check(questions, {d: normalize(t) for d, t in docs.items()}, chunks):
        raise SystemExit("labels have errors: run py -3 -m evals.check_labels")

    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    rows = [question_row(q, ranked) for q, ranked in zip(questions, rank(model, questions, chunks))]
    report = {
        "provenance": provenance(model, (ROOT / "questions.toml").read_bytes()),
        "summary": summarize(rows),
        "questions": rows,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, s in report["summary"].items():
        print(split, "  ".join(f"{m} {v}" for m, v in s.items()))
    for r in rows:
        print(r["id"], r["split"], "unanswerable" if r["scores"] is None else f"fact ranks {r['fact_ranks']}")
    print(f"wrote {REPORT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
