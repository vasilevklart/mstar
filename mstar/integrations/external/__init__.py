"""Serve a graph node from someone else's engine over an OpenAI-compatible API.

For vendors that expose no device to run M* on (Cerebras, Groq, SambaNova and
similar), the only way in is as a client: a node whose tokens come from a
streaming ``/v1/completions`` or ``/v1/chat/completions`` endpoint instead of a
forward pass. ``docs/design/external_engine_node.md`` has the design; this
package is its first part, the parts that need no worker or graph-runtime
change:

- ``config``: the ``external:`` block of a node group, read and checked.
- ``openai_stream``: one streaming request, its chunks, usage and failures.
- ``streams``: the per-request streams a node keeps in flight, polled without
  blocking, cancelled on request removal, retried only before the first output.

Requires ``aiohttp``. Nothing outside this package imports it.
"""
