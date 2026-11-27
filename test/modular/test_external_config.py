"""The ``external:`` block of a node group (``mstar.integrations.external.config``):
what it reads, and what it refuses at startup rather than per request."""
from __future__ import annotations

import pytest

from mstar.integrations.external.config import ExternalConfigError, external_endpoints


def _config(external, ranks=(1,), **group):
    return {"node_groups": [
        {"node_names": ["vision"], "ranks": [0]},
        {"node_names": ["LLM"], "ranks": list(ranks), "external": external, **group},
    ]}


BLOCK = {"base_url": "https://api.example.com/v1/", "model": "qwen-3.8-27b", "api_key_env": "EXAMPLE_KEY"}


def test_config_reads_the_external_block():
    endpoints = external_endpoints(_config(BLOCK))
    assert list(endpoints) == ["LLM"]
    endpoint = endpoints["LLM"]
    assert endpoint.base_url == "https://api.example.com/v1", "a trailing slash is dropped"
    assert (endpoint.adapter, endpoint.max_in_flight, endpoint.retries) == ("openai_completions", 16, 2)
    assert external_endpoints({"node_groups": [{"node_names": ["LLM"], "ranks": [0]}]}) == {}


@pytest.mark.parametrize("config, match", [
    (_config({**BLOCK, "api_key": "sk-123"}), "api_key_env"),
    (_config({**BLOCK, "max_inflight": 4}), "unknown keys"),
    (_config({**BLOCK, "adapter": "grpc"}), "adapter"),
    (_config({**BLOCK, "base_url": "api.example.com"}), "http"),
    (_config({**BLOCK, "max_in_flight": 0}), "max_in_flight"),
    (_config({**BLOCK, "read_timeout_s": True}), "read_timeout_s"),
    (_config({k: v for k, v in BLOCK.items() if k != "model"}), "model"),
    (_config(BLOCK, ranks=(0,)), "both external and local"),
    (_config(BLOCK, ranks=(1, 2)), "one rank"),
    (_config(BLOCK, tp_size=2), "one rank"),
    (_config(BLOCK, ranks=()), "needs `ranks`"),
])
def test_config_refusals(config, match):
    with pytest.raises(ExternalConfigError, match=match):
        external_endpoints(config)


def test_api_key_env_must_be_set(monkeypatch):
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    endpoint = external_endpoints(_config(BLOCK))["LLM"]
    with pytest.raises(ExternalConfigError, match="EXAMPLE_KEY"):
        endpoint.api_key()
    monkeypatch.setenv("EXAMPLE_KEY", "k")
    assert endpoint.api_key() == "k"
