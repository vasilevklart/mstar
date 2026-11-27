"""A fake OpenAI-compatible streaming endpoint on loopback, for the external-engine node's tests
(``test_external_streams.py``, ``test_external_resource.py``). No network: it listens on 127.0.0.1 only."""
from __future__ import annotations

import asyncio
import json
import threading
import time

from aiohttp import web


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


def _slow(attempt, body):
    """200 one-token events 20 ms apart: a stream still running when a test cancels it."""
    steps = []
    for i in range(200):
        steps += [_event("x", ids=[i]), 0.02]
    return steps + [_event("", finish="stop"), DONE]


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
