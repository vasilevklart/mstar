---
orphan: true
---

# External-engine nodes (design note, partly implemented)

Status: proposed. The client half and the node's resource are in `mstar/integrations/external/` (the `external:` config block, the per-request streams, and `ExternalStream`, which runs them through the resource lifecycle), with tests against a fake endpoint. Nothing in the worker, the engine or the graph runtime builds them yet; the last section lists what does.

## The problem

Some inference hardware can't run M\* at all. Cerebras, Groq and SambaNova expose no device that PyTorch or a plugin can target. Their compilers and runtimes are their own, and the only public way in is an OpenAI-compatible HTTP API ([Cerebras](https://inference-docs.cerebras.ai/resources/openai.md), [Groq](https://console.groq.com/docs/openai), [SambaNova](https://docs.sambanova.ai/docs/en/models/sambacloud-models)). So they can't be a `Platform`: there is no torch device to resolve one from (`mstar/platforms/__init__.py`).

They can still serve one node of an M\* graph. A text LLM node takes token ids and returns token ids, and these endpoints take token ids and return tokens. Cerebras's `/v1/completions` takes "an array of tokens" as the prompt ([completions](https://inference-docs.cerebras.ai/api-reference/completions.md)), so M\* keeps its own chat template and tokenizer. The node's upstream and downstream (a vision tower before it, a TTS model after it) stay on local GPUs and see no difference.

## Where it plugs in

The node keeps its name and its place in the model's graph. Only the deployment config changes (invariant 8):

```yaml
node_groups:
  - node_names: [vision_encoder]
    ranks: [0]
  - node_names: [LLM]
    ranks: [1]
    external:
      base_url: https://api.cerebras.ai/v1
      model: qwen-3.8-27b
      adapter: openai_completions      # token ids in; openai_chat sends messages
      api_key_env: CEREBRAS_API_KEY    # the key never goes in the YAML
      max_in_flight: 16
rank_devices: {1: cpu}                 # not on main yet: see "What is left"
```

Rank 1 is an ordinary worker. It runs the usual control mesh, graph runtime and micro-scheduler, and receives the node's inputs over the usual tensor transport. What changes is what that worker builds for `LLM`:

| Layer | Local node | External node |
| --- | --- | --- |
| Submodule | the model's own (`Model.get_submodule`), with weights | one generic `ExternalSubmodule`, no weights, chosen because the config names the node `external` |
| Resources | the model's declared state: paged KV, sampler, … | one resource, `ExternalStream`, that owns the request's HTTP stream |
| Forward | the model's layers | a copy of the tokens the resource landed for the row |
| Device | a GPU | none; the worker runs on CPU |

The model's graph walks, `process_prompt`, tokenizer and output handling don't change. The node emits its tokens on the same edges the local node would, so they reach the client through the same `emit_to_client` path and the same detokenizer.

### Why a resource, and not I/O in the forward

Invariant 3 rules out I/O in a forward, and invariant 1 says a schedulable condition must be backpressure, not an exception. The resource lifecycle already has a hook for "data for this row is arriving from elsewhere": `Resource.admit_retrieve` is how a decode node waits for a KV transfer from a prefill worker, and the micro-scheduler asks it on every scan of a graph-ready row (`MicroScheduler._scan_ready` → `Admission.check_ready` → `Engine.check_ready` → `StepRunner.admit_retrieve`). The remote stream is the same thing with a longer tail. Each request lifecycle call maps onto one call in `mstar/integrations/external/streams.py`:

| Resource hook (`mstar/engine/resources/base.py`) | `ExternalStream` (`resource.py`) |
| --- | --- |
| `ask` / `reserve` / `grew_since` | the `max_in_flight` budget: past it the request queues, as it would for KV pages, and starts when one ahead of it is removed |
| `commit` of a step that carries the prompt (`ExternalStep.prompts`) | `start(rid, prompt_ids, settings)` |
| `admit_retrieve` | `poll(rid)`: not ready until a token can be emitted; a terminal outcome when the stream failed |
| `plan` | the token each row emits and whether it is the row's last (`ExternalPlan`) |
| `remove_request` | `cancel(rid)` |

The requests run on an event loop of their own thread, so the worker's loop never waits on the network. That thread calls back (`ExternalStreams.on_change`, `ExternalStream.bind_wake`) when output lands or a request ends.

A token is emitted only once the token after it has landed or the stream has ended. Endpoints send the finish reason in an event of their own after the last token, so this is what lets the step that emits the last token know it is the last, and the stop check end the loop there instead of on a step with nothing to emit. It costs one event of latency per token.

### What it streams

One API call yields one stream of many tokens. In M\* terms that is an autoregressive loop on the node, the same `prefill` then `decode` walks the model already has, where a decode step runs as soon as at least one token has landed for the row. The graph runtime already expresses this: a loop node that isn't ready doesn't step, and `check_stop` ends the loop. From reading the runtime and the scheduler, no change to `rust/` is needed for this (invariant 5).

Endpoints send several tokens per event: Cerebras says so ([designing for Cerebras](https://inference-docs.cerebras.ai/resources/designing-for-cerebras.md)), and the client lands every token of an event in order. The first version emits one token per decode step, which keeps the client path unchanged. Its cost is one worker step per token. At a few thousand tokens a second per user, a step's CPU overhead may cap the node below the endpoint's rate. Emitting a row's landed tokens as a span in one step, as speculative decoding does (`spanned_outputs`), is the fix if the measurement says so.

In `openai_completions` mode the node needs token ids back, not text. The client reads `token_ids` off each choice when an endpoint sends them; recent vLLM releases do when the request sets `return_token_ids`, so the per-vendor table below also has to say which request field asks for them. Cerebras documents a `return_raw_tokens` flag but not whether it returns ids, so the first live run has to probe it. When an endpoint returns only text, as the OpenAI API itself does, the node either re-tokenizes with M\*'s tokenizer, which can split differently at chunk boundaries, or runs in `openai_chat` mode at a graph boundary, where the endpoint owns the template and the output is text for the client.

## Cancellation, errors and usage

**Cancellation.** A client abort, a drain or a removal reaches the worker as today, and `Engine.remove_request` calls every resource's `remove_request`. The external resource cancels the stream there. That shuts the HTTP connection, which is the only cancel an OpenAI-compatible server understands, also while the endpoint is still silent before its first token. Anything that lands afterwards is dropped. Whether a vendor bills tokens it generated after the disconnect is not documented by any of the three.

**Errors.** Two cases, split by whether output has gone downstream:

- Before the first token, a 429, a 5xx or a dropped connection is retried, honouring `Retry-After`, up to `retries` times. While it waits the row is just not ready. A 4xx other than 408, 409 and 429 fails at once.
- After the first token nothing is retried. The stream can't be resumed, and its prefix has already reached the client. An error event in the stream, a stream that ends without a finish reason, or a timeout between events (`read_timeout_s`) fails the request.

A failure reaches the client through the existing path. `admit_retrieve` returns a terminal outcome, and `Admission.check_ready` fails the request with the resource's message, which the worker sends to the conductor as `FAIL_REQUESTS`. A failure wins over tokens that landed before it and are still waiting to be emitted: emitting them would end the loop on the last one as if the endpoint had finished, and the client would get a short answer with no error. The node's state lives on the endpoint, where M\* can neither copy nor recompute it. So M\* can rebuild the request from its token ids only before the first output, and the node declares no rebuild policy.

**Usage.** The prompt tokens are M\*'s own count, since M\* tokenized the prompt, and the completion tokens are the tokens the node emitted. The client sees the same usage it would from a local node. The endpoint's own usage (`stream_options.include_usage`) is kept as a per-request metric. Its cached prompt tokens (`prompt_tokens_details.cached_tokens`) are the endpoint's prefix-cache hits, which no local resource sees. A completion count that disagrees with M\*'s means the tokenizers drifted, and is logged.

**Settings.** Temperature, top-p, seed, stop strings and max tokens map onto request fields. A setting the endpoint can't honour is refused when the request is ingested, never dropped. `n > 1` is the first case (`openai_stream.unsupported`). A per-vendor table follows from the vendors' compatibility pages: Groq, for one, rejects `logprobs` and `logit_bias` ([compatibility](https://console.groq.com/docs/openai)).

## What the node refuses at startup

- A node that consumes or produces hidden states. These APIs take text, token ids and images, so a node fed embeddings by another node (an omni model's talker, an ASR model's LLM fed encoder states) can't move out.
- A node group with parallel axes or more than one rank, and a rank shared with a local node group. The endpoint does its own sharding, and a worker waiting on a remote stream has no device for a local node's forward.
- An API key written into the YAML.

`mstar/integrations/external/config.py` already checks the last two.

## Data leaving the deployment

An external node sends its prompts to a third party. That is decided per node in the deployment config and nowhere else. No default, environment variable or fallback turns it on, and the key is read from the environment when a request is built.

## What is left

Built: `config.py` (the block and its refusals), `openai_stream.py` (one streaming request, its chunks, usage and failures), `streams.py` (per-request streams, polling, cancel, retry before the first output, the in-flight budget, a callback when output lands) and `resource.py` (`ExternalStream`, the table above), with tests in `test/modular/test_external_config.py`, `test_external_streams.py` and `test_external_resource.py`. The resource's tests drive the real `StepRunner` against a fake endpoint on loopback: one token per step with the last flagged, the endpoint's usage, a mid-stream error as a terminal outcome with the endpoint's message, removal closing the connection (also while the endpoint is silent), and the budget.

Still to build, in order:

1. **Where the prompt reaches the resource.** The local LLM's prefill step emits the first token (`media_walks` sends `new_token` from it to the client). The external node's prompt step starts the stream when it commits, so it has no token yet. Either the node's prefill step emits nothing and the first token comes from the first decode step, which needs a step that sends nothing on the walk's `new_token` edge (or a span of zero tokens, `spanned_outputs`), or the prompt ids reach the resource before the prefill step, so that step waits for the first token like any decode step. Today a resource sees no input tensors before its step, and `CurrentForwardPassInfo` carries no prompt ids. This is the open design question, and it decides the shape of the next item.
2. **The swap.** Where `EngineManager.build` calls `Model.get_submodule` and `serving_resources` builds the model's declared state, build `ExternalSubmodule` and an `ExternalStreamSpec` for a node `external_endpoints` names, and add the startup refusal for hidden-state edges. The submodule declares an `ExternalStep` (prompt ids and `SamplingSettings` from the request on the prompt step), its forward copies `ExternalPlan.token_ids` to the walk's output, its stop check reads `ExternalPlan.last`, and it sets `enable_async_scheduling=False`: there is nothing to speculate. Settings the endpoint can't honour are refused at ingest (`openai_stream.unsupported`).
3. **Waking the worker.** Bind `ExternalStream.bind_wake` to the worker's `EventWakeup.signal`, through `EngineResourceInfo` or after `load_model`.
4. **A GPU-less rank.** `deployment_workers` gives every rank the accelerator's device type (`mstar/conductor/workers.py`). Letting `rank_devices` say `cpu` for a rank means a worker on a GPU host can claim no GPU. A CPU-only deployment doesn't need it.
5. **A CPU end-to-end test**: a deployment whose LLM node is external, against the fake endpoint, through the conductor, the worker and the data worker, text out on the OpenAI chat route: streaming, a mid-stream error as a clean error, a client abort reaching the endpoint, and usage. It needs a tokenizer available offline, and a Linux host: the worker's `EventWakeup` is an `os.eventfd`, which macOS doesn't have.
6. **A live A/B** of the same model served locally and from an endpoint, with token-level agreement at temperature 0 from logprobs, TTFT and tokens per second from `benchmark.runner`.

Later: rate limits as admission (requests and tokens per minute beside `max_in_flight`), a local and an external replica of one node with overflow routing in `mstar/conductor/routing.py`, and adapters for endpoints that serve speech models.
