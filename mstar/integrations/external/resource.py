"""The external node's one resource: each request's stream to the endpoint, run through the resource lifecycle.

=========================  ====================================================================================
Hook                       What it does here
=========================  ====================================================================================
``ask`` / ``reserve``      the in-flight budget (``max_in_flight``): past it a request waits to start, as one
                           waits for KV pages, and starts once a request ahead of it is removed
``commit``                 starts the stream of every row whose step carries a prompt (``ExternalStep.prompts``)
``admit_retrieve``         a row with a stream is ready once it has a token to emit; a failed stream is a
                           terminal outcome, which fails the request with the endpoint's message
``plan``                   the token each row emits this step, and whether it is the row's last
``remove_request``         cancels the stream: its connection closes, which is how the endpoint learns
=========================  ====================================================================================

A token is emitted only once the token after it has landed or the stream has ended, so the step that emits the
last token knows it is the last (``ExternalPlan.last``) and the node's stop check can end the loop on it. An
endpoint sends its finish reason in an event of its own after the last token, so without that hold-back the loop
would need a step with nothing to emit. It costs one event of latency per token.

Nothing here waits on the network. The streams run on their own thread (`ExternalStreams`), which calls ``wake``
when output lands so a sleeping worker rescans.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    AdmitOutcome,
    AdmitRuntimeError,
    ResourceStep,
    StepContext,
)
from mstar.integrations.external.config import ExternalEndpoint
from mstar.integrations.external.openai_stream import SamplingSettings, Usage
from mstar.integrations.external.streams import ExternalStreams

logger = logging.getLogger(__name__)

# not ready: the row's next token hasn't landed
_WAITING = AdmitOutcome(ok=True, ready=False)
# ``ExternalPlan.token_ids`` for a row that emits nothing this step (its prompt step)
NO_TOKEN = -1


@dataclass
class ExternalStreamSpec(NodeResourceSpec):
    """The stream resource of the nodes an ``external:`` block names, one per endpoint."""

    endpoint: ExternalEndpoint

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.endpoint, ExternalEndpoint):
            raise TypeError(f"{self.resource_key}: an external stream needs an ExternalEndpoint")

    @property
    def resource_class(self) -> type[Resource]:
        return ExternalStream


@dataclass(frozen=True)
class ExternalStep(ResourceStep):
    """A step's work for the stream resource. A row in ``prompts`` starts its stream when the step commits, with
    its ``settings``; every other row of the step emits its next landed token."""

    prompts: Mapping[Any, Sequence[int] | str | list[dict]] = field(default_factory=dict, hash=False)
    settings: Mapping[Any, SamplingSettings] = field(default_factory=dict, hash=False)


class ExternalPlan(NamedTuple):
    """What a step's rows emit, in ``ctx.request_ids`` order, on the host."""

    # the token each row emits; NO_TOKEN for a row that only starts its stream
    token_ids: torch.Tensor
    # whether that token is the row's last: the stream has ended and nothing else is left
    last: tuple[bool, ...]


@dataclass(slots=True)
class _Row:
    """One request's stream, as the resource sees it."""

    # landed and not yet emitted
    pending: deque = field(default_factory=deque)
    emitted: int = 0
    ended: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None
    error: str | None = None


class ExternalStream(Resource):
    """Owns every request's stream to one endpoint. Built per spec, so nodes that name one endpoint share its
    budget."""

    def __init__(self, endpoint: ExternalEndpoint, wake: Callable[[], None] | None = None):
        self.endpoint = endpoint
        self._wake = wake
        self.streams = ExternalStreams(endpoint, on_change=self._on_change)
        # rid -> its stream's state, from the commit that started it until it is removed
        self._rows: dict[Any, _Row] = {}
        # rids holding a slot of the budget, from `reserve` until removed
        self._reserved: set = set()
        # slots given back so far, the mark `grew_since` compares against
        self._released = 0
        # the worker's main thread scans readiness and removes requests while its step thread admits, plans and
        # commits, so the per-row state is guarded
        self._lock = threading.Lock()

    @classmethod
    def build(cls, spec: ExternalStreamSpec, info: EngineResourceInfo) -> ExternalStream:
        del info  # no device, no process group: the endpoint does its own sharding
        return cls(spec.endpoint)

    def bind_wake(self, wake: Callable[[], None]) -> None:
        """Have the streams thread call ``wake`` when output lands, so the worker rescans instead of sleeping."""
        self._wake = wake

    def _on_change(self) -> None:
        wake = self._wake
        if wake is not None:
            wake()

    def can_capture(self) -> bool:
        # a step's tokens come from the network, not a forward a graph could replay
        return False

    # Budget

    def ask(self, rid, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        return None if rid in self._reserved else ASKS

    def reserve(self, rid, node_name: str, graph_walk: str, published) -> object | None:
        if rid in self._reserved:
            return None
        if len(self._reserved) >= self.endpoint.max_in_flight:
            return self._released
        self._reserved.add(rid)
        return None

    def unreserve(self, rid) -> None:
        self._release(rid)

    def grew_since(self, mark: object) -> bool:
        return self._released != mark

    def _release(self, rid) -> None:
        if rid in self._reserved:
            self._reserved.discard(rid)
            self._released += 1

    # Readiness

    def admit_retrieve(self, rid, node_name: str, graph_walk: str, published) -> AdmitOutcome:
        with self._lock:
            row = self._rows.get(rid)
            if row is None:
                # no stream yet: this is the row's prompt step, which starts one
                return ADMIT_OK
            self._land(rid, row)
            if row.error is not None:
                # Fails the request even with tokens left to emit: emitting them would end the loop on the last
                # one as if the endpoint had finished, and the client would never see the error.
                return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                    f"external endpoint {self.endpoint.base_url}: {row.error}", frozenset({rid}),
                ))
            if row.pending and (len(row.pending) > 1 or row.ended):
                return ADMIT_OK
            if row.ended:
                # the stop check ends a row's loop on the step that emits its last token, so a row that comes back
                # with nothing left had no tokens at all
                return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                    f"external endpoint {self.endpoint.base_url} ended the stream without a token "
                    f"(finish reason {row.finish_reason})", frozenset({rid}),
                ))
            return _WAITING

    def _land(self, rid, row: _Row) -> None:
        """Move what landed since the last poll onto the row."""
        if row.ended:
            return
        landed = self.streams.poll(rid)
        if landed is None:
            # nothing to poll: the stream ended and was polled already, or never started
            row.ended = True
            return
        if landed.text and not landed.token_ids:
            row.error = (
                "the endpoint returned text without token ids; this node needs ids back (vLLM: return_token_ids), "
                "or the openai_chat adapter at a graph boundary"
            )
            row.ended = True
            self.streams.cancel(rid)
            return
        row.pending.extend(landed.token_ids)
        if landed.usage is not None:
            row.usage = landed.usage
        if landed.finish_reason is not None:
            row.finish_reason = landed.finish_reason
        if landed.error is not None:
            row.error = str(landed.error)
        row.ended = landed.done

    # Step lifecycle

    def plan(self, step: ExternalStep, ctx: StepContext) -> ExternalPlan:
        prompts = step.prompts
        ids, last = [], []
        with self._lock:
            for rid in ctx.request_ids:
                if rid in prompts:
                    ids.append(NO_TOKEN)
                    last.append(False)
                    continue
                row = self._rows[rid]
                ids.append(row.pending[0])
                last.append(row.ended and row.error is None and len(row.pending) == 1)
        return ExternalPlan(torch.tensor(ids, dtype=torch.long), tuple(last))

    def commit(self, step: ExternalStep, ctx: StepContext) -> None:
        prompts = step.prompts
        with self._lock:
            for rid in ctx.request_ids:
                if rid in prompts:
                    if rid in self._rows:
                        raise RuntimeError(f"request {rid} already has an external stream")
                    # started under the lock, so no readiness scan sees the row before its stream exists
                    self.streams.start(rid, prompts[rid], step.settings[rid])
                    self._rows[rid] = _Row()
                else:
                    row = self._rows[rid]
                    row.pending.popleft()
                    row.emitted += 1

    # Request lifecycle

    def usage(self, rid) -> Usage | None:
        """The endpoint's own count for ``rid``, when it sent one."""
        row = self._rows.get(rid)
        return None if row is None else row.usage

    def emitted(self, rid) -> int:
        row = self._rows.get(rid)
        return 0 if row is None else row.emitted

    def remove_request(self, rid) -> None:
        self.streams.cancel(rid)
        with self._lock:
            row = self._rows.pop(rid, None)
            self._release(rid)
        if row is not None and row.ended and row.error is None and row.usage is not None:
            if row.usage.completion_tokens != row.emitted + len(row.pending):
                # the endpoint counts differently from the ids it sent: its tokenizer or its stop handling drifted
                logger.warning(
                    "external request %s: endpoint counted %d completion tokens, sent %d ids",
                    rid, row.usage.completion_tokens, row.emitted + len(row.pending),
                )

    def cleanup(self) -> None:
        self.streams.close()
