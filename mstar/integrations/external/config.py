"""The ``external:`` block of a node group: which endpoint serves its nodes.

::

    node_groups:
      - node_names: [LLM]
        ranks: [1]
        external:
          base_url: https://api.example.com/v1
          model: qwen-3.8-27b
          adapter: openai_completions      # or openai_chat
          api_key_env: EXAMPLE_API_KEY     # the key itself never goes in YAML
          max_in_flight: 16

The rank still names a worker, which hosts the node without a device of its
own. Sending a node's prompts off the deployment is a decision made here, per
node, and nowhere else.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from mstar.distributed.base import node_group_axis_sizes

ADAPTERS = ("openai_completions", "openai_chat")


class ExternalConfigError(ValueError):
    """An ``external:`` block that can't be served as written."""


@dataclass(frozen=True)
class ExternalEndpoint:
    """One endpoint and the limits a node keeps against it."""

    base_url: str
    model: str
    adapter: str = "openai_completions"
    api_key_env: str | None = None
    # Requests open at once: the node's admission budget. One past it waits.
    max_in_flight: int = 16
    connect_timeout_s: float = 10.0
    # Longest gap between two stream events before the request fails.
    read_timeout_s: float = 60.0
    # Attempts after the first, for a 429, a 5xx or a dropped connection
    # before any output. Never after: the stream can't be resumed.
    retries: int = 2

    @classmethod
    def from_block(cls, where: str, block: Any) -> ExternalEndpoint:
        if not isinstance(block, Mapping):
            raise ExternalConfigError(f"{where}: `external` must be a mapping; got {block!r}")
        if "api_key" in block:
            raise ExternalConfigError(
                f"{where}: `external` names an api_key. Put the key in an environment variable and name "
                "the variable in `api_key_env`, so it stays out of configs and logs."
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - known)
        if unknown:
            raise ExternalConfigError(f"{where}: `external` has unknown keys {unknown}; it takes {sorted(known)}")
        for key in ("base_url", "model"):
            if not isinstance(block.get(key), str) or not block[key]:
                raise ExternalConfigError(f"{where}: `external.{key}` must be a non-empty string")
        if not block["base_url"].startswith(("http://", "https://")):
            raise ExternalConfigError(f"{where}: `external.base_url` must be an http(s) URL; got {block['base_url']!r}")
        endpoint = cls(**{**block, "base_url": block["base_url"].rstrip("/")})
        if endpoint.adapter not in ADAPTERS:
            raise ExternalConfigError(
                f"{where}: `external.adapter` must be one of {ADAPTERS}; got {endpoint.adapter!r}"
            )
        if not isinstance(endpoint.max_in_flight, int) or endpoint.max_in_flight < 1:
            raise ExternalConfigError(f"{where}: `external.max_in_flight` must be an integer >= 1")
        if not isinstance(endpoint.retries, int) or endpoint.retries < 0:
            raise ExternalConfigError(f"{where}: `external.retries` must be an integer >= 0")
        for key in ("connect_timeout_s", "read_timeout_s"):
            value = getattr(endpoint, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ExternalConfigError(f"{where}: `external.{key}` must be a number > 0")
        return endpoint

    def api_key(self) -> str | None:
        """The key from ``api_key_env``, read when a request is built. A named variable that is unset is an error,
        not an anonymous request."""
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ExternalConfigError(f"`external.api_key_env` names {self.api_key_env}, which is not set")
        return key


def external_endpoints(config: Mapping[str, Any]) -> dict[str, ExternalEndpoint]:
    """Node name -> endpoint, for every node group with an ``external`` block.

    An external group's ranks host nothing else: a worker that waits on a remote stream has no device to run a local
    node's forward on."""
    groups = config.get("node_groups") or []
    endpoints: dict[str, ExternalEndpoint] = {}
    external_ranks: set[int] = set()
    local_ranks: set[int] = set()
    for i, group in enumerate(groups):
        if not isinstance(group, Mapping):
            continue
        ranks = set(group.get("ranks") or ())
        if "external" not in group:
            local_ranks |= ranks
            continue
        where = f"node_groups[{i}]"
        if not ranks:
            raise ExternalConfigError(f"{where}: an external node group still needs `ranks`, the worker hosting it")
        sizes = node_group_axis_sizes(group)
        if any(size != 1 for size in sizes.values()) or len(ranks) != 1:
            raise ExternalConfigError(
                f"{where}: an external node group runs on one rank with no parallel axes; the endpoint does its own "
                f"sharding (got ranks {sorted(ranks)}, {sizes})"
            )
        endpoint = ExternalEndpoint.from_block(where, group["external"])
        for name in group.get("node_names") or ():
            endpoints[name] = endpoint
        external_ranks |= ranks
    shared = sorted(external_ranks & local_ranks)
    if shared:
        raise ExternalConfigError(f"ranks {shared} host both external and local node groups")
    return endpoints
