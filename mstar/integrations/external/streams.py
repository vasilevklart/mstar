"""The streams a node keeps open against its endpoint, one per request.

The worker's loop must never wait on the network, so the requests run on an
event loop of their own, on its own thread, and the node only ever asks what
has landed (``poll``). The calls map onto a resource's request lifecycle:

- ``has_room``: the admission budget. A request past ``max_in_flight`` waits
  to start, as a step waits for KV pages, rather than failing.
- ``start``: send the request.
- ``poll``: the output that landed since the last poll, without blocking; a
  row with nothing new isn't ready this pass.
- ``cancel``: the request was aborted, drained or removed. Its connection is
  closed and anything still arriving is dropped.

A failure before the first output is retried (``ExternalEndpoint.retries``);
one after it ends the request, since a stream can't be resumed and its prefix
has already gone downstream.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import aiohttp

from mstar.integrations.external.config import ExternalEndpoint
from mstar.integrations.external.openai_stream import (
    SamplingSettings,
    StreamError,
    Usage,
    open_stream,
)
from mstar.utils.threads import Thread

logger = logging.getLogger(__name__)

# Backoff before a retry, when the endpoint names no Retry-After; doubled per attempt.
RETRY_BASE_S = 0.25
RETRY_CAP_S = 8.0


@dataclass(frozen=True)
class Landed:
    """What a poll returns for one request: output since the last poll, and how the request ended if it has."""

    text: str = ""
    token_ids: tuple[int, ...] = ()
    finished: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None
    error: StreamError | None = None

    @property
    def done(self) -> bool:
        return self.finished or self.error is not None


@dataclass(slots=True)
class _Stream:
    task: asyncio.Future | None = None
    text: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    finished: bool = False
    error: StreamError | None = None
    attempts: int = 0


class ExternalStreams:
    """In-flight requests to one endpoint. ``start``, ``poll``, ``cancel`` and ``has_room`` are called from the
    worker's thread; the requests themselves run on this object's loop thread.

    ``on_change`` is called from the loop thread whenever a request has something new to poll (output, or its end),
    so a worker sleeping until there is work wakes up for it. It must not block."""

    def __init__(self, endpoint: ExternalEndpoint, on_change: Callable[[], None] | None = None):
        self.endpoint = endpoint
        self.on_change = on_change
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._loop = asyncio.new_event_loop()
        self._session: aiohttp.ClientSession | None = None
        self._thread = Thread(target=self._loop.run_forever, name="mstar-external-streams", daemon=True)
        self._thread.start()

    # Worker-side calls

    def has_room(self) -> bool:
        with self._lock:
            return len(self._streams) < self.endpoint.max_in_flight

    def in_flight(self) -> int:
        with self._lock:
            return len(self._streams)

    def start(self, rid: str, prompt: Sequence[int] | str | list[dict], settings: SamplingSettings) -> None:
        with self._lock:
            if rid in self._streams:
                raise ValueError(f"request {rid} already has a stream")
            if len(self._streams) >= self.endpoint.max_in_flight:
                raise RuntimeError(f"no room for request {rid}: {self.endpoint.max_in_flight} in flight; ask has_room")
            stream = self._streams[rid] = _Stream()
        stream.task = asyncio.run_coroutine_threadsafe(self._run(rid, stream, prompt, settings), self._loop)

    def poll(self, rid: str) -> Landed | None:
        """Output landed since the last poll, or None for a request this object doesn't know (never started, or
        already cancelled or finished and polled). A request that ended is forgotten once its end is polled."""
        with self._lock:
            stream = self._streams.get(rid)
            if stream is None:
                return None
            landed = Landed(
                "".join(stream.text), tuple(stream.token_ids), stream.finished, stream.finish_reason, stream.usage,
                stream.error,
            )
            stream.text.clear()
            stream.token_ids.clear()
            if landed.done:
                del self._streams[rid]
            return landed

    def cancel(self, rid: str) -> bool:
        """Close the request's connection and forget it. False when there was nothing to cancel."""
        with self._lock:
            stream = self._streams.pop(rid, None)
            self._changed.notify_all()
        if stream is None:
            return False
        if stream.task is not None:
            stream.task.cancel()
        return True

    def wait(self, rid: str, timeout: float) -> bool:
        """Block until ``rid`` has something to poll or is gone. For tests and tools; the worker polls."""
        def ready() -> bool:
            stream = self._streams.get(rid)
            if stream is None or stream.finished or stream.error is not None:
                return True
            return bool(stream.text or stream.token_ids)

        with self._changed:
            return self._changed.wait_for(ready, timeout)

    def close(self) -> None:
        with self._lock:
            rids = list(self._streams)
        for rid in rids:
            self.cancel(rid)
        if self._session is not None:
            asyncio.run_coroutine_threadsafe(self._session.close(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # Loop-side

    def _publish(self, rid: str, stream: _Stream, **changes) -> bool:
        """Apply ``changes`` unless the request was cancelled meanwhile. False once it was."""
        with self._lock:
            if self._streams.get(rid) is not stream:
                return False
            for key, value in changes.items():
                if key == "text":
                    stream.text.append(value)
                elif key == "token_ids":
                    stream.token_ids.extend(value)
                else:
                    setattr(stream, key, value)
            self._changed.notify_all()
        if self.on_change is not None:
            self.on_change()
        return True

    async def _run(self, rid: str, stream: _Stream, prompt, settings: SamplingSettings) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        while True:
            stream.attempts += 1
            emitted = 0
            try:
                async for chunk in open_stream(self._session, self.endpoint, prompt, settings):
                    emitted += bool(chunk.text or chunk.token_ids)
                    changes = {}
                    if chunk.text:
                        changes["text"] = chunk.text
                    if chunk.token_ids:
                        changes["token_ids"] = chunk.token_ids
                    if chunk.usage is not None:
                        changes["usage"] = chunk.usage
                    if chunk.finish_reason is not None:
                        changes["finish_reason"] = chunk.finish_reason
                    if changes and not self._publish(rid, stream, **changes):
                        return
                self._publish(rid, stream, finished=True)
                return
            except StreamError as failure:
                failure.emitted = max(failure.emitted, emitted)
                if failure.retryable and failure.emitted == 0 and stream.attempts <= self.endpoint.retries:
                    delay = failure.retry_after
                    if delay is None:
                        delay = random.uniform(0.5, 1.0) * min(RETRY_CAP_S, RETRY_BASE_S * 2 ** (stream.attempts - 1))
                    logger.info("external request %s: %s; retry %d in %.2fs", rid, failure, stream.attempts, delay)
                    await asyncio.sleep(min(delay, RETRY_CAP_S))
                    continue
                self._publish(rid, stream, error=failure)
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:  # a bug here must still end the request, not leave it waiting forever
                logger.exception("external request %s failed", rid)
                failure = StreamError(f"{type(error).__name__}: {error}", False, emitted=emitted)
                self._publish(rid, stream, error=failure)
                return
