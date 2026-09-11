"""Deterministic test doubles shared across test modules."""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np


class FakeEmbeddingModel:
    """Small deterministic embedder with controllable semantic directions."""

    _terms = ("california", "neural", "apple", "weather")

    def encode(
        self, texts: Iterable[str], normalize_embeddings: bool = True
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.array(
                [text.lower().count(term) for term in self._terms], dtype=float
            )
            if not vector.any():
                vector[0] = 1.0
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype=float)


def chunk(
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """One streamed chunk, shaped like an OpenAI ChatCompletionChunk."""
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)]
    )


def text_chunk(text: str) -> SimpleNamespace:
    return chunk(content=text)


def thought_chunk(text: str) -> SimpleNamespace:
    return chunk(reasoning=text)


def call_fragment(
    index: int = 0,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """A piece of a tool call: any field may be absent in any given chunk."""
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def call_chunk(name: str, index: int = 0, **args: Any) -> SimpleNamespace:
    """A chunk carrying one whole tool call, arguments already serialized."""
    return chunk(
        tool_calls=[
            call_fragment(
                index=index,
                id=f"call_{index}_{name}",
                name=name,
                arguments=json.dumps(args),
            )
        ]
    )


class UnscriptedModelCall(BaseException):
    """The agent asked for a turn the test did not script.

    Deliberately a BaseException: agent_loop catches Exception, so an ordinary error
    would come back as a stream_reset plus an error event and leave the test green on a
    model call nobody planned.
    """


class ScriptedClient:
    """OpenAI client stand-in replaying one prepared turn per completions.create call.

    `turns` is a list of turns; each turn is the list of chunks that the stream
    delivers. Running out of turns raises UnscriptedModelCall, so an unplanned extra
    model call fails the test instead of silently repeating the last answer. `sent` holds
    a deep copy of the messages per call, because the agent mutates one list in place.
    """

    def __init__(self, turns: Iterable[list[SimpleNamespace]]) -> None:
        self._turns = iter(turns)
        self.sent: list[list[dict]] = []
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    @property
    def calls(self) -> int:
        return len(self.sent)

    def tool_replies(self, call_index: int) -> list[dict]:
        """The tool messages the agent added before model call `call_index`."""
        return [m for m in self.sent[call_index] if m["role"] == "tool"]

    async def _create(self, **kwargs: Any):
        self.sent.append(copy.deepcopy(kwargs["messages"]))
        self.requests.append(kwargs)
        try:
            chunks = next(self._turns)
        except StopIteration:
            raise UnscriptedModelCall(f"model call #{len(self.sent)} was not scripted") from None

        async def stream():
            for item in chunks:
                yield item

        return stream()


class LoopingClient(ScriptedClient):
    """Never stops asking for tools, so only the agent's own limit can end the run."""

    def __init__(self, *chunks: SimpleNamespace) -> None:
        super().__init__(turns=itertools.repeat(list(chunks)))
