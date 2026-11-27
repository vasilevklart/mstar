"""One streaming request to an OpenAI-compatible endpoint.

``open_stream`` yields a ``StreamChunk`` per event that carries output, and a
last one with the finish reason and usage. Everything that goes wrong raises
``StreamError``, which says whether a retry may help (``retryable``) and how
much output had been yielded before it (``emitted``): only a request that has
emitted nothing can be retried, since the endpoint can't resume a stream.

Cancelling the consuming task closes the connection, which is the only cancel
an OpenAI-compatible server understands.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import aiohttp

from mstar.integrations.external.config import ExternalEndpoint

# Statuses worth another attempt before any output: rate limits, overload, a
# gateway in front of the endpoint restarting.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class Usage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    # Prompt tokens the endpoint served from its own prefix cache, when it says.
    cached_tokens: int = 0


@dataclass(frozen=True)
class StreamChunk:
    """New output from one event. Token ids are set when the endpoint returns them; ``text`` always is."""

    text: str = ""
    token_ids: tuple[int, ...] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass
class StreamError(Exception):
    message: str
    retryable: bool
    status: int | None = None
    # Chunks with output yielded before the failure.
    emitted: int = 0
    # Seconds the endpoint asked to wait (Retry-After), when it did.
    retry_after: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = f"HTTP {self.status}: " if self.status is not None else ""
        return f"{status}{self.message}"


@dataclass(frozen=True)
class SamplingSettings:
    """The request settings an endpoint can take. Anything else M* supports is refused before a request is sent
    (``unsupported``), never dropped."""

    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()


def unsupported(settings: dict[str, Any]) -> str | None:
    """Why a request's settings can't be sent to an endpoint, or None. Every OpenAI-compatible vendor we know of
    takes one choice per request."""
    if settings.get("n", 1) != 1:
        return "an external node returns one choice per request (n=1)"
    return None


def request_body(
    endpoint: ExternalEndpoint, prompt: Sequence[int] | str | list[dict], settings: SamplingSettings,
) -> tuple[str, dict[str, Any]]:
    """The path and JSON body for one streaming request. ``openai_completions`` takes token ids, so M* keeps its own
    template and tokenizer; ``openai_chat`` takes messages and the endpoint applies its template."""
    body: dict[str, Any] = {
        "model": endpoint.model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": settings.max_tokens,
    }
    for key in ("temperature", "top_p", "seed"):
        value = getattr(settings, key)
        if value is not None:
            body[key] = value
    if settings.stop:
        body["stop"] = list(settings.stop)
    if endpoint.adapter == "openai_completions":
        if isinstance(prompt, str):
            body["prompt"] = prompt
        else:
            body["prompt"] = [int(t) for t in prompt]
        return "/completions", body
    if not isinstance(prompt, list) or not all(isinstance(m, dict) for m in prompt):
        raise TypeError("openai_chat takes a list of messages")
    body["messages"] = prompt
    return "/chat/completions", body


def _usage(raw: dict[str, Any] | None) -> Usage | None:
    if not raw:
        return None
    details = raw.get("prompt_tokens_details") or {}
    return Usage(
        int(raw.get("prompt_tokens") or 0),
        int(raw.get("completion_tokens") or 0),
        int(details.get("cached_tokens") or 0),
    )


def parse_event(data: dict[str, Any]) -> StreamChunk | None:
    """One SSE ``data:`` object as a chunk, or None for an event with nothing in it (a role-only first delta)."""
    if "error" in data:
        error = data["error"]
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise StreamError(f"endpoint error mid-stream: {message}", retryable=False, details={"error": error})
    usage = _usage(data.get("usage"))
    choices = data.get("choices") or []
    if not choices:
        return StreamChunk(usage=usage) if usage is not None else None
    choice = choices[0]
    delta = choice.get("delta")
    text = (delta.get("content") or "") if isinstance(delta, dict) else (choice.get("text") or "")
    ids = choice.get("token_ids")
    chunk = StreamChunk(
        text=text,
        token_ids=tuple(int(t) for t in ids) if ids else None,
        finish_reason=choice.get("finish_reason"),
        usage=usage,
    )
    if not chunk.text and chunk.token_ids is None and chunk.finish_reason is None and chunk.usage is None:
        return None
    return chunk


def _retry_after(headers) -> float | None:
    value = headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


async def open_stream(
    session: aiohttp.ClientSession,
    endpoint: ExternalEndpoint,
    prompt: Sequence[int] | str | list[dict],
    settings: SamplingSettings,
) -> AsyncIterator[StreamChunk]:
    """Send one request and yield its chunks; the last one carries ``finish_reason`` (and ``usage`` when the
    endpoint sends it, which may be on a chunk of its own after that)."""
    path, body = request_body(endpoint, prompt, settings)
    headers = {"Accept": "text/event-stream"}
    key = endpoint.api_key()
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    timeout = aiohttp.ClientTimeout(sock_connect=endpoint.connect_timeout_s, sock_read=endpoint.read_timeout_s)
    emitted = 0
    finished = False
    try:
        async with session.post(endpoint.base_url + path, json=body, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                text = (await response.text())[:500]
                raise StreamError(
                    text or response.reason or "request refused",
                    retryable=response.status in RETRYABLE_STATUS,
                    status=response.status,
                    retry_after=_retry_after(response.headers),
                )
            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue  # blank separators, comments, event: lines
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    if not finished:
                        raise StreamError("stream ended with [DONE] before a finish reason", False, emitted=emitted)
                    return
                try:
                    chunk = parse_event(json.loads(payload))
                except StreamError as failure:
                    failure.emitted = emitted
                    raise
                except (ValueError, TypeError, AttributeError) as error:
                    raise StreamError(f"unreadable stream event: {error}", False, emitted=emitted) from error
                if chunk is None:
                    continue
                if chunk.text or chunk.token_ids:
                    emitted += 1
                finished = finished or chunk.finish_reason is not None
                yield chunk
    except StreamError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        raise StreamError(
            f"connection failed: {type(error).__name__}: {error}", retryable=emitted == 0, emitted=emitted,
        ) from error
    if not finished:
        raise StreamError("stream closed before a finish reason", retryable=emitted == 0, emitted=emitted)
