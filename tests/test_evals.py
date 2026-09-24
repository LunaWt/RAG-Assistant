from evals.check_labels import check
from evals.corpus import normalize
from evals.retrieval import fact_ranks, score, summarize

DOC = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi iota kappa lambda mu"
DOCS = {"d": normalize(DOC)}
CHUNKS = [{"doc": "d", "i": 0, "text": DOC[:40]}, {"doc": "d", "i": 1, "text": DOC[30:]}]


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
            question("q4", quotes=["iota kappa lambda mu"]),
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


def test_recall_counts_facts_and_mrr_takes_the_first_relevant_chunk():
    ranked = [
        {"doc": "other", "text": "Six times faster decoding"},
        {"doc": "d", "text": "... six times  FASTER decoding ..."},
        {"doc": "d", "text": "filler"},
        {"doc": "d", "text": "filler"},
        {"doc": "d", "text": "prefill 2.9x at 1M"},
    ]
    evidence = [
        {"doc": "d", "quotes": ["six times faster decoding"]},
        {"doc": "d", "quotes": ["prefill 2.9x at 1M"]},
    ]
    ranks = fact_ranks(ranked, evidence)
    assert ranks == [2, 5]
    assert score(ranks) == {"recall@1": 0, "recall@3": 0.5, "recall@5": 1, "recall@10": 1, "rr": 0.5}


def test_unanswerable_questions_stay_out_of_the_means():
    hit = score([1])
    miss = score([None])
    rows = [
        {"split": "dev", "scores": hit},
        {"split": "dev", "scores": miss},
        {"split": "dev", "scores": None},
        {"split": "test", "scores": hit},
    ]
    summary = summarize(rows)
    assert summary["dev"]["answerable"] == 2
    assert summary["dev"]["mrr"] == 0.5
    assert summary["dev"]["recall@3"] == 0.5
