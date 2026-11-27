"""The external-engine node's client half (``mstar.integrations.external``): the
per-request streams it keeps against an OpenAI-compatible endpoint, driven
against a fake endpoint on loopback, with no network.

Covers streaming (several tokens per event, usage, the request body), a failure
mid-stream (no retry once output went out), a stream cut short, retries before
the first output, cancellation (the connection closes and nothing more lands,
also while the endpoint is still silent), and the in-flight budget."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from mstar.integrations.external.config import ExternalEndpoint  # noqa: E402
from mstar.integrations.external.openai_stream import SamplingSettings, unsupported  # noqa: E402
from mstar.integrations.external.streams import ExternalStreams  # noqa: E402

pytestmark = pytest.mark.timeout(60)


def _event(text="", ids=None, finish=None, usage=None, chat=False):
    choice = {"index": 0, "finish_reason": finish}
    if chat:
        choice["delta"] = {"content": text}
    else:
        choice["text"] = text
    if ids is not None:
        choice["token_ids"] = ids
    event = {"id": "cmpl-1", "object": "text_completion", "choices": [choice]}
    if usage is not None:
        event["usage"] = usage
    return event


USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}}
USAGE_EVENT = {"id": "cmpl-1", "choices": [], "usage": USAGE}
DONE = "[DONE]"


class FakeEndpoint:
    """An OpenAI-compatible streaming server on 127.0.0.1, on a thread of its own. ``script(attempt, body)`` returns
    what to answer: an int status (with optional headers) or a list of steps, each an event dict, ``DONE``, or a
    float to sleep. A list that doesn't end in ``DONE`` closes the stream short."""

    def __init__(self, script):
        self.script = script
        self.requests: list[tuple[str, dict, dict]] = []
        self.disconnected = threading.Event()
        self._loop = asyncio.new_event_loop()
        started = threading.Event()
        self._thread = threading.Thread(target=self._serve, args=(started,), daemon=True)
        self._thread.start()
        assert started.wait(10)

    def _serve(self, started):
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_post("/v1/completions", self._handle)
        app.router.add_post("/v1/chat/completions", self._handle)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        started.set()
        self._loop.run_forever()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def _handle(self, request):
        body = await request.json()
        self.requests.append((request.path, body, dict(request.headers)))
        answer = self.script(len(self.requests), body)
        if isinstance(answer, tuple):
            status, headers = answer
            return web.json_response({"error": {"message": "slow down"}}, status=status, headers=headers)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            for step in answer:
                if isinstance(step, float):
                    # a silent endpoint (a long prefill) still notices the client leaving
                    deadline = time.monotonic() + step
                    while time.monotonic() < deadline:
                        if request.transport is None or request.transport.is_closing():
                            raise ConnectionResetError("client went away")
                        await asyncio.sleep(0.005)
                    continue
                payload = step if step == DONE else json.dumps(step)
                await response.write(f"data: {payload}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            self.disconnected.set()
            raise
        return response

    def close(self):
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


@pytest.fixture
def serve():
    made = []

    def make(script, **endpoint):
        server = FakeEndpoint(script)
        streams = ExternalStreams(ExternalEndpoint(base_url=server.base_url, model="m", **endpoint))
        made.append((server, streams))
        return server, streams

    yield make
    for server, streams in made:
        streams.close()
        server.close()


def drain(streams: ExternalStreams, rid: str, timeout: float = 10.0):
    """Poll until the request ends, the way a node would across steps; returns (texts, ids, last Landed)."""
    texts, ids = [], []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        streams.wait(rid, timeout=0.5)
        landed = streams.poll(rid)
        assert landed is not None, "the request disappeared before it ended"
        if landed.text:
            texts.append(landed.text)
        ids.extend(landed.token_ids)
        if landed.done:
            return texts, ids, landed
    raise AssertionError(f"request {rid} didn't end in {timeout}s")


SETTINGS = SamplingSettings(max_tokens=8, temperature=0.0, seed=7)


# --- streaming ---------------------------------------------------------------------------------------------------


def test_streams_tokens_usage_and_finish(serve, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sekrit")
    script = lambda attempt, body: [  # noqa: E731
        _event("Hel", ids=[11, 12]), 0.01, _event("lo", ids=[13]), _event("", finish="stop"), USAGE_EVENT, DONE,
    ]
    server, streams = serve(script, api_key_env="FAKE_KEY")
    streams.start("r1", [1, 2, 3, 4, 5], SETTINGS)
    texts, ids, last = drain(streams, "r1")

    assert "".join(texts) == "Hello"
    assert ids == [11, 12, 13], "a multi-token event lands every token, in order"
    assert last.finished and last.error is None and last.finish_reason == "stop"
    assert (last.usage.prompt_tokens, last.usage.completion_tokens, last.usage.cached_tokens) == (5, 3, 2)
    assert streams.poll("r1") is None, "an ended request is forgotten once its end is polled"
    assert streams.in_flight() == 0

    [(path, body, headers)] = server.requests
    assert path == "/v1/completions"
    assert body["prompt"] == [1, 2, 3, 4, 5], "tokens mode sends M*'s own token ids"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert (body["model"], body["max_tokens"], body["temperature"], body["seed"]) == ("m", 8, 0.0, 7)
    assert "top_p" not in body, "an unset setting is left to the endpoint, not sent as null"
    assert headers["Authorization"] == "Bearer sekrit"


def test_chat_adapter_sends_messages_and_reads_deltas(serve):
    script = lambda attempt, body: [  # noqa: E731
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        _event("Hi", chat=True), _event(" there", chat=True), _event("", finish="length", chat=True), DONE,
    ]
    server, streams = serve(script, adapter="openai_chat")
    streams.start("r1", [{"role": "user", "content": "hello"}], SETTINGS)
    texts, ids, last = drain(streams, "r1")
    assert "".join(texts) == "Hi there" and ids == []
    assert last.finish_reason == "length" and last.usage is None
    [(path, body, _)] = server.requests
    assert path == "/v1/chat/completions" and body["messages"] == [{"role": "user", "content": "hello"}]


# --- failures ----------------------------------------------------------------------------------------------------


def test_error_mid_stream_fails_the_request_without_retry(serve):
    script = lambda attempt, body: [  # noqa: E731
        _event("a", ids=[1]), {"error": {"message": "model overloaded", "type": "server_error"}}, DONE,
    ]
    server, streams = serve(script, retries=3)
    streams.start("r1", [1], SETTINGS)
    texts, ids, last = drain(streams, "r1")
    assert ids == [1], "output that landed before the failure is still delivered"
    assert last.error is not None and not last.finished
    assert "model overloaded" in str(last.error)
    assert last.error.emitted == 1
    assert len(server.requests) == 1, "a stream that already produced output is never retried"


def test_stream_cut_short_is_a_failure(serve):
    server, streams = serve(lambda attempt, body: [_event("a", ids=[1])], retries=3)
    streams.start("r1", [1], SETTINGS)
    _, ids, last = drain(streams, "r1")
    assert ids == [1]
    assert last.error is not None and "before a finish reason" in str(last.error)
    assert len(server.requests) == 1


def test_rate_limit_before_output_is_retried(serve):
    def script(attempt, body):
        if attempt == 1:
            return 429, {"Retry-After": "0"}
        return [_event("ok", ids=[5], finish="stop"), DONE]

    server, streams = serve(script, retries=2)
    streams.start("r1", [1], SETTINGS)
    _, ids, last = drain(streams, "r1")
    assert last.finished and last.error is None and ids == [5]
    assert len(server.requests) == 2


def test_rate_limit_past_the_retries_fails_with_the_status(serve):
    server, streams = serve(lambda attempt, body: (429, {"Retry-After": "0"}), retries=1)
    streams.start("r1", [1], SETTINGS)
    _, _, last = drain(streams, "r1")
    assert last.error is not None and last.error.status == 429 and last.error.retryable
    assert len(server.requests) == 2


def test_client_error_is_not_retried(serve):
    server, streams = serve(lambda attempt, body: (400, {}), retries=3)
    streams.start("r1", [1], SETTINGS)
    _, _, last = drain(streams, "r1")
    assert last.error.status == 400 and not last.error.retryable
    assert len(server.requests) == 1


# --- cancellation and budget -------------------------------------------------------------------------------------


def _slow(attempt, body):
    steps = []
    for i in range(200):
        steps += [_event("x", ids=[i]), 0.02]
    return steps + [_event("", finish="stop"), DONE]


def test_cancel_closes_the_connection_and_drops_later_output(serve):
    server, streams = serve(_slow)
    streams.start("r1", [1], SETTINGS)
    assert streams.wait("r1", timeout=10)
    first = streams.poll("r1")
    assert first is not None and first.token_ids and not first.done

    assert streams.cancel("r1")
    assert server.disconnected.wait(5), "the endpoint saw the connection close"
    assert streams.poll("r1") is None
    assert streams.in_flight() == 0 and streams.has_room()
    assert not streams.cancel("r1"), "a second cancel finds nothing"
    time.sleep(0.1)
    assert streams.poll("r1") is None, "output still in flight after the cancel is dropped"


def test_cancel_while_the_endpoint_is_silent(serve):
    """Before the first token (a long prefill) there is no event to notice the cancel on, so the cancel itself has
    to close the connection."""
    server, streams = serve(lambda attempt, body: [30.0, _event("late", ids=[1], finish="stop"), DONE])
    streams.start("r1", [1], SETTINGS)
    deadline = time.monotonic() + 10
    while not server.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.requests
    assert streams.cancel("r1")
    assert server.disconnected.wait(5), "the endpoint saw the connection close while it was still silent"
    assert streams.poll("r1") is None and streams.has_room()


def test_in_flight_budget(serve):
    server, streams = serve(_slow, max_in_flight=1)
    streams.start("r1", [1], SETTINGS)
    assert not streams.has_room()
    with pytest.raises(RuntimeError, match="no room"):
        streams.start("r2", [1], SETTINGS)
    with pytest.raises(ValueError, match="already has a stream"):
        streams.start("r1", [1], SETTINGS)
    streams.cancel("r1")
    assert streams.has_room()
    streams.start("r2", [1], SETTINGS)
    assert streams.wait("r2", timeout=10)


def test_unsupported_settings():
    assert unsupported({"n": 1}) is None
    assert "n=1" in unsupported({"n": 2})
