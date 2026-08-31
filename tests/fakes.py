"""Deterministic test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
from google.genai import types


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


def chunk(*parts: types.Part) -> SimpleNamespace:
    """One streamed chunk carrying the given parts."""
    return SimpleNamespace(parts=list(parts))


def text_part(text: str, thought: bool = False) -> types.Part:
    return types.Part(text=text, thought=thought or None)


def call_part(name: str, **args: Any) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class ScriptedChat:
    """Gemini chat stand-in replaying one prepared turn per send_message_stream call.

    `turns` is a list of turns; each turn is the list of chunks that stream
    delivers. Running out of turns raises StopIteration, so an unplanned extra
    model call fails the test instead of silently repeating the last answer.
    `sent` records what the agent sent, including tool-result parts.
    """

    def __init__(self, turns: list[list[SimpleNamespace]]) -> None:
        self._turns = iter(turns)
        self.sent: list[Any] = []

    async def send_message_stream(self, message: Any):
        self.sent.append(message)
        chunks = next(self._turns)

        async def stream():
            for item in chunks:
                yield item

        return stream()


class LoopingChat:
    """Gemini chat stand-in that never stops asking for tools.

    Every turn replies with the same part, so a model that keeps calling tools
    forever can only be stopped by the agent's own iteration limit. `calls`
    counts model round-trips, including the opening turn and the final nudge.
    """

    def __init__(self, part: types.Part) -> None:
        self._part = part
        self.calls = 0

    async def send_message_stream(self, message: Any):
        self.calls += 1

        async def stream():
            yield chunk(self._part)

        return stream()


def scripted_client(chat: Any) -> SimpleNamespace:
    """Stand-in for genai.Client whose chats.create() always hands back `chat`.

    `chat` is any object exposing send_message_stream: ScriptedChat, LoopingChat
    or a one-off fake defined inside a test.
    """
    return SimpleNamespace(
        aio=SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    )