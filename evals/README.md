# Retrieval evaluation (M11)

## Corpus

Four arXiv papers, pinned by version and PDF hash in `corpus.toml`. `corpus/` is gitignored,
because the licences (arXiv non-exclusive, CC BY-NC-ND) do not allow redistributing the text.
`py -3 -m evals.build_corpus` fetches any missing PDF from arXiv, checks its hash, transcribes
it through the production vision path, chunks it with the production `chunk_size` / `overlap`,
and writes `corpus/snapshot.json` with the hash of every transcript and of the chunk file.
A transcript is reused only while the PDF hash, both vision models and the prompt are
unchanged; changing any of them re-transcribes. The VLM is not deterministic, so a fresh machine
gets slightly different text, and `check_labels` reports every quote that no longer matches.
`check_labels` refuses a transcript or chunk file that differs from the snapshot.

## Labels: `questions.toml`

```toml
[[question]]
id = "q01"
split = "dev"                       # dev: may be looked at while tuning; test: only for the report
text = "How much does HySparse2 cut prefill FLOPs at 1M context?"
answer = "About 5x."                  # key points a grader looks for, compared by meaning

[[question.evidence]]               # one fact the answer needs
doc = "hysparse2"
quotes = ['''a span copied from corpus/md/hysparse2.md''', '''the same fact stated elsewhere''']
```

A question with no `[[question.evidence]]` is unanswerable from the corpus; its `answer` says
why, and a correct response abstains. `answer` is never matched as a string.
Copy quotes from `corpus/md/<doc>.md`, not from the PDF: the transcript carries Markdown
(`**bold**`, LaTeX), and a quote only matches the text the retriever actually indexes.

`py -3 -m evals.check_labels` checks every quote: 20–200 characters, present exactly once in
its document, and inside a single chunk. The 200 cap keeps quotes shorter than the
300-character overlap; a phrase that recurs is rejected, because every chunk repeating it
would count as a hit.

## Definitions

- **Relevant chunk.** A retrieved chunk is relevant to a fact if it belongs to the fact's
  document and contains one of that fact's quotes, compared case-insensitively with whitespace
  collapsed. The same sentence in another paper does not count.
- **recall@k** for one question: facts with a relevant chunk in the top k ÷ all its facts.
  Reported as the mean over answerable questions.
- **MRR**: the mean over answerable questions of 1 / rank of the first chunk relevant to any
  fact, with 0 when none appears in the top 10.
- **Unanswerable questions** are excluded from both denominators. Counting them as 0 would
  punish the retriever for a question with no right chunk, and counting them as 1 would reward
  it for free. They are scored in answer evaluation, where abstaining is the correct outcome.

## Retrieval report

`py -3 -m evals.retrieval` refuses to run on a corpus that differs from the snapshot or on
labels that fail `check_labels`. It embeds every chunk and question with the production
embedder (`settings.embedding_model`, `normalize_embeddings=True`), ranks the chunks of all
four papers by exact cosine similarity, and scores the top 10 by the definitions above. Chroma
in production ranks by L2 distance over the same normalized vectors, which orders identically
(‖a − b‖² = 2 − 2·a·b), but through an approximate index. Production passes the top 3 to the
model, so recall@3 is the number that describes the live app.

It writes `results/retrieval.json`: `summary` holds the means per split, `questions` the rank
of each fact (null when it is outside the top 10) and the ten chunk ids retrieved, and
`provenance` the git commit, whether `app/` or `evals/` had uncommitted changes, the hashes of
`questions.toml`, the snapshot and the chunk file, the embedding model with its Hub revision,
the device and the library versions. The report is committed; the transcripts it was computed
from stay local, so an exact rerun needs the same `corpus/` that the snapshot hash names.
