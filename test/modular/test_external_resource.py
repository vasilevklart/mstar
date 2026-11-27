"""The external node's resource (``mstar.integrations.external.resource``), driven through the real ``StepRunner``
lifecycle against a fake OpenAI-compatible endpoint on loopback, with no network.

A request is admitted against the in-flight budget, its prompt step starts the stream at commit, and each later
step is ready once a token has landed for it and emits one token. Covers streaming (one token per step, the last
one flagged, the endpoint's usage kept), the wake-up when output lands, a mid-stream error as a terminal outcome
with the endpoint's message, removal cancelling the stream (also while the endpoint is silent), and the budget."""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("aiohttp")
from external_endpoint import DONE, USAGE_EVENT, FakeEndpoint, _event, _slow  # noqa: E402

from mstar.engine.resources import StepContext, StepRunner, SubmoduleStep  # noqa: E402
from mstar.integrations.external.config import ExternalEndpoint  # noqa: E402
from mstar.integrations.external.openai_stream import SamplingSettings  # noqa: E402
from mstar.integrations.external.resource import (  # noqa: E402
    NO_TOKEN,
    ExternalStep,
    ExternalStream,
    ExternalStreamSpec,
)

pytestmark = pytest.mark.timeout(60)

NODE = "LLM"
KEY = "external"
SETTINGS = SamplingSettings(max_tokens=8, temperature=0.0)
PROMPT = [101, 102, 103]


class Node:
    """The external resource behind a real ``StepRunner``, stepped the way an engine steps a node: readiness
    through ``admit_retrieve``, then admit, plan and commit."""

    def __init__(self, server: FakeEndpoint, **endpoint):
        self.woken = threading.Event()
        spec = ExternalStreamSpec(KEY, {NODE}, ExternalEndpoint(base_url=server.base_url, model="m", **endpoint))
        self.resource = ExternalStream.build(spec, None)
        self.resource.bind_wake(self.woken.set)
        self.runner = StepRunner({KEY: self.resource}, node_resources={NODE: [KEY]})

    def ready(self, rid, walk="decode"):
        return self.runner.admit_retrieve(rid, NODE, walk)

    def step(self, rids, walk="decode", prompts=None):
        prompts = prompts or {}
        step = SubmoduleStep(steps={KEY: ExternalStep(prompts=prompts, settings=dict.fromkeys(prompts, SETTINGS))})
        step.set_ctx(StepContext(request_ids=list(rids), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(step).ok
        plan = self.runner.plan(step)[KEY]
        self.runner.commit(step)
        return plan

    def start(self, rid):
        """Admit ``rid`` and run its prompt step, which emits nothing and starts the stream."""
        assert self.ready(rid, "prefill_text").ready
        plan = self.step([rid], "prefill_text", prompts={rid: PROMPT})
        assert plan.token_ids.tolist() == [NO_TOKEN] and plan.last == (False,)

    def wait_ready(self, rid, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            outcome = self.ready(rid)
            if not outcome.ok or outcome.ready:
                return outcome
            time.sleep(0.005)
        raise AssertionError(f"request {rid} wasn't ready in {timeout}s")

    def drain(self, rid):
        """Step ``rid`` until a step flags its last token; returns the tokens."""
        tokens = []
        while True:
            outcome = self.wait_ready(rid)
            assert outcome.ok, outcome.reason
            plan = self.step([rid])
            tokens += plan.token_ids.tolist()
            if plan.last[0]:
                return tokens


@pytest.fixture
def node():
    made = []

    def make(script, **endpoint):
        server = FakeEndpoint(script)
        made.append((server, Node(server, **endpoint)))
        return made[-1]

    yield make
    for server, n in made:
        n.resource.cleanup()
        server.close()


# --- streaming ---------------------------------------------------------------------------------------------------


def test_one_token_per_step_the_last_flagged_and_usage_kept(node):
    script = lambda attempt, body: [  # noqa: E731
        _event("Hel", ids=[11, 12]), _event("lo", ids=[13]), _event("", finish="stop"), USAGE_EVENT, DONE,
    ]
    server, n = node(script)
    rid = 7  # the worker's handle for the request
    n.start(rid)

    assert n.drain(rid) == [11, 12, 13], "every landed token is emitted, in order, one per step"
    assert n.woken.is_set(), "output landing wakes the worker"
    assert n.resource.emitted(rid) == 3
    usage = n.resource.usage(rid)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens) == (5, 3, 2)

    [(path, body, _)] = server.requests
    assert path == "/v1/completions" and body["prompt"] == PROMPT, "the prompt step's ids are what is sent"
    assert (body["max_tokens"], body["temperature"]) == (8, 0.0)


def test_a_token_waits_for_the_one_after_it(node):
    """The step that emits the last token has to know it is the last, and the finish reason comes in an event of
    its own, so a token is held until the next one lands or the stream ends."""
    script = lambda attempt, body: [  # noqa: E731
        _event("a", ids=[1]), 0.3, _event("b", ids=[2]), _event("", finish="stop"), DONE,
    ]
    server, n = node(script)
    n.start(1)
    deadline = time.monotonic() + 10
    while not n.resource._rows[1].pending and time.monotonic() < deadline:
        assert not n.ready(1).ready
        time.sleep(0.005)
    assert n.resource._rows[1].pending, "the first token landed"
    assert not n.ready(1).ready, "one landed token, stream still open: not ready"
    assert n.drain(1) == [1, 2]


# --- failures ----------------------------------------------------------------------------------------------------


def test_error_mid_stream_fails_the_request_with_the_endpoints_message(node):
    script = lambda attempt, body: [  # noqa: E731
        _event("a", ids=[1]), _event("b", ids=[2]), 0.2, {"error": {"message": "model overloaded"}}, DONE,
    ]
    server, n = node(script, retries=3)
    n.start(1)
    assert n.wait_ready(1).ok
    assert n.step([1]).token_ids.tolist() == [1]

    outcome = n.wait_ready(1)
    assert not outcome.ok and outcome.failed_resource == KEY
    assert outcome.reason.terminal, "the worker fails the request rather than rescanning it"
    assert "model overloaded" in outcome.reason.message and server.base_url in outcome.reason.message
    assert len(server.requests) == 1, "nothing is retried once output went out"


def test_text_without_token_ids_fails_the_request(node):
    server, n = node(lambda attempt, body: [_event("hi"), _event("", finish="stop"), DONE])
    n.start(1)
    outcome = n.wait_ready(1)
    assert not outcome.ok and outcome.reason.terminal
    assert "without token ids" in outcome.reason.message


# --- cancellation and budget -------------------------------------------------------------------------------------


def test_removing_the_request_closes_the_endpoints_connection(node):
    server, n = node(_slow)
    n.start(1)
    assert n.wait_ready(1).ok
    n.step([1])
    n.runner.remove_request(1)
    assert server.disconnected.wait(5), "the endpoint saw the connection close"
    assert n.resource.streams.in_flight() == 0


def test_removal_while_the_endpoint_is_silent(node):
    server, n = node(lambda attempt, body: [30.0, _event("late", ids=[1], finish="stop"), DONE])
    n.start(1)
    deadline = time.monotonic() + 10
    while not server.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.requests and not n.ready(1).ready
    n.runner.remove_request(1)
    assert server.disconnected.wait(5), "the endpoint saw the connection close before its first token"


def test_past_max_in_flight_a_request_waits_until_one_is_removed(node):
    server, n = node(_slow, max_in_flight=1)
    n.start(1)
    waiting = n.ready(2, "prefill_text")
    assert waiting.queued, "past the budget a request waits for room, as for KV pages, rather than failing"
    assert n.runner.admit_waiting(lambda rid, node_name: None) == []

    n.runner.remove_request(1)
    assert n.runner.admit_waiting(lambda rid, node_name: None) == [(NODE, 2)]
    n.start(2)
    assert n.wait_ready(2).ok
