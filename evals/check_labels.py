import sys
import tomllib
from collections import Counter

from evals.corpus import ROOT, load_chunks, load_verified_docs, normalize

MIN_QUOTE, MAX_QUOTE = 20, 200
SPLITS = {"dev", "test"}


def load_questions() -> list[dict]:
    with open(ROOT / "questions.toml", "rb") as f:
        return tomllib.load(f).get("question", [])


def quote_error(quote: str, doc: str, docs: dict[str, str], chunks: list[tuple[str, str]]) -> str | None:
    norm = normalize(quote)
    if not MIN_QUOTE <= len(norm) <= MAX_QUOTE:
        return f"quote is {len(norm)} chars, keep it {MIN_QUOTE}-{MAX_QUOTE}"
    first = docs[doc].find(norm)
    if first == -1:
        return f"not found in {doc}.md"
    # A phrase that recurs makes every chunk repeating it count as a hit. Searching from
    # first + 1 also catches a repeat that overlaps the first match, which str.count skips.
    if docs[doc].find(norm, first + 1) != -1:
        return f"too generic, more than once in {doc}.md"
    if not any(d == doc and norm in text for d, text in chunks):
        return "split across a chunk boundary"
    return None


def check(questions: list[dict], docs: dict[str, str], chunks: list[dict]) -> list[str]:
    errors = []
    ids = Counter(q.get("id") for q in questions)
    errors += [f"{i}: duplicate id" for i, n in ids.items() if n > 1]
    normalized = [(c["doc"], normalize(c["text"])) for c in chunks]
    for q in questions:
        qid = q.get("id", "?")
        if not q.get("text"):
            errors.append(f"{qid}: no text")
        if q.get("split") not in SPLITS:
            errors.append(f"{qid}: split must be one of {sorted(SPLITS)}")
        for n, fact in enumerate(q.get("evidence", []), 1):
            doc = fact.get("doc")
            if doc not in docs:
                errors.append(f"{qid} fact {n}: unknown doc {doc!r}")
                continue
            if not fact.get("quotes"):
                errors.append(f"{qid} fact {n}: no quotes")
            for quote in fact.get("quotes", []):
                error = quote_error(quote, doc, docs, normalized)
                if error:
                    errors.append(f"{qid} fact {n}: {error}: {quote[:60]!r}")
    return errors


def main() -> None:
    questions = load_questions()
    docs = {doc_id: normalize(text) for doc_id, text in load_verified_docs().items()}
    errors = check(questions, docs, load_chunks())
    answerable = [q for q in questions if q.get("evidence")]
    splits = Counter(q.get("split") for q in questions)
    print(f"{len(questions)} questions: {len(answerable)} answerable, "
          f"{len(questions) - len(answerable)} unanswerable; "
          f"dev {splits['dev']}, test {splits['test']}")
    for e in errors:
        print("ERROR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
