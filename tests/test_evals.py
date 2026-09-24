from evals.check_labels import check
from evals.corpus import normalize

DOC = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu lambda mu lambda"
DOCS = {"d": normalize(DOC)}
REPEATED = "lambda mu lambda mu lambda"
CHUNKS = [{"doc": "d", "i": 0, "text": DOC[:40]}, {"doc": "d", "i": 1, "text": DOC[30:]}] + [
    {"doc": "d", "i": i, "text": REPEATED} for i in (2, 3, 4)
]


def question(qid="q1", quotes=("beta gamma delta epsilon",), split="dev"):
    evidence = [{"doc": "d", "quotes": list(quotes)}] if quotes else []
    return {"id": qid, "split": split, "text": "?", "evidence": evidence}


def test_quote_inside_one_chunk_passes_whitespace_and_case_blind():
    assert check([question(quotes=["BETA  gamma\ndelta epsilon"])], DOCS, CHUNKS) == []


def test_unanswerable_question_needs_no_evidence():
    assert check([question(quotes=())], DOCS, CHUNKS) == []


def test_each_label_error_is_reported():
    errors = check(
        [
            question("q1", quotes=["beta gamma delta epsilon omega"]),
            question("q2", quotes=["epsilon zeta eta theta"]),
            question("q3", split="train"),
            question("q4", quotes=["lambda mu lambda mu lambda"]),
            question("q3"),
        ],
        DOCS,
        CHUNKS,
    )
    assert any("q1" in e and "not found" in e for e in errors)
    assert any("q2" in e and "chunk boundary" in e for e in errors)
    assert any("q3" in e and "split" in e for e in errors)
    assert any("q3" in e and "duplicate" in e for e in errors)
    assert any("q4" in e and "too generic" in e for e in errors)
