"""Torch stand-ins for FlashInfer's two GDN kernels on a CPU tensor, with their signatures
and results (the pool's K-last ``[HV, V, K]`` state), so the GDN resource's paths run in a
test without a GPU. Token by token in fp32: a reference, not a fast path."""

import torch
import torch.nn.functional as F


def _step(state, q, k, v, alpha, beta, scale):
    """One token of the gated delta rule over every head: ``state`` ``[HV, V, K]`` fp32, ``q``
    and ``k`` already at the value heads. Returns the output ``[HV, V]`` and the new state."""
    state = state * alpha[:, None, None]
    delta = v - torch.einsum("hvk,hk->hv", state, k)
    state = state + beta[:, None, None] * delta[:, :, None] * k[:, None, :]
    return torch.einsum("hvk,hk->hv", state, q) * scale, state


def _at_value_heads(x: torch.Tensor, value_heads: int) -> torch.Tensor:
    # a key head serves the value heads after it in order (GVA), as the model's TP shards assume
    return x.repeat_interleave(value_heads // x.shape[-2], dim=-2)


def chunk_gated_delta_rule(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None,
    use_qk_l2norm_in_kernel=False, output_state=None, state_indices=None, **checkpoints,
):
    """FlashInfer's chunked prefill: packed ``q``/``k`` ``[T, H, K]``, ``v`` ``[T, HV, V]``, the decay
    ``g`` exponentiated and ``beta`` ``[T, HV]``, each sequence of ``cu_seqlens`` from its
    ``initial_state`` row. Returns the output ``[T, HV, V]`` (in ``q``'s dtype) and the final states
    ``[N, HV, V, K]`` fp32. Refuses a zero-length sequence, on which the kernel deadlocks.

    With ``state_indices`` (a slot per sequence), ``initial_state`` and ``output_state`` are pools:
    sequence ``i`` starts from row ``state_indices[i]`` and its final state goes to that row of
    ``output_state``, which is returned in place of the packed finals. As in the kernel, every
    sequence reads its row, whatever the row holds, and a slot outside the pool (a negative one
    included) is refused, since the kernel has no skip for it. Every row is read before any is
    written, as the kernel's tiles each load their state before storing it."""
    assert not checkpoints, "the CPU GDN prefill keeps no state checkpoints"
    assert not use_qk_l2norm_in_kernel, "FlashInfer's chunk kernel ignores use_qk_l2norm_in_kernel"
    bounds = cu_seqlens.tolist()
    assert all(b > a for a, b in zip(bounds[:-1], bounds[1:], strict=True)), (
        f"a zero-length sequence in {bounds}, on which the chunked kernel deadlocks"
    )
    heads, dim_k = v.shape[1], q.shape[-1]
    rows = range(len(bounds) - 1)
    if state_indices is not None:
        slots = state_indices.tolist()
        assert len(slots) == len(rows), f"{len(slots)} state_indices for {len(rows)} sequences"
        assert output_state is not None, "with state_indices the finals go into output_state, the pool"
        assert all(0 <= s < output_state.shape[0] for s in slots), (
            f"a slot outside the pool in {slots}: the kernel has no skip for it"
        )
    else:
        slots = list(rows)
    starts = [
        initial_state[s].float().clone() if initial_state is not None else v.new_zeros(
            heads, v.shape[2], dim_k, dtype=torch.float32,
        )
        for s in slots
    ]
    scale = dim_k ** -0.5 if scale is None else scale
    qf, kf = (_at_value_heads(t.float(), heads) for t in (q, k))
    vf, gf, bf = v.float(), g.float(), beta.float()
    out = torch.zeros(v.shape, dtype=torch.float32)
    finals = []
    for state, (start, end) in zip(starts, zip(bounds[:-1], bounds[1:], strict=True), strict=True):
        for t in range(start, end):
            out[t], state = _step(state, qf[t], kf[t], vf[t], gf[t], bf[t], scale)
        finals.append(state)
    if state_indices is None:
        return out.to(q.dtype), torch.stack(finals) if output_final_state else None
    for slot, final in zip(slots, finals, strict=True):
        output_state[slot] = final.to(output_state.dtype)
    return out.to(q.dtype), output_state


def gated_delta_rule_decode_pretranspose(
    q, k, v, state=None, A_log=None, a=None, dt_bias=None, b=None, scale=None, output=None,
    initial_state=None, initial_state_indices=None, use_qk_l2norm=True,
):
    """FlashInfer's decode over the pool in place: ``q``/``k`` ``[B, 1, H, K]``, ``v`` ``[B, 1, HV,
    V]``, the gates raw (``a``/``b`` ``[B, 1, HV]``, decay ``exp(-exp(A_log) softplus(a +
    dt_bias))``, rate ``sigmoid(b)``), row ``i`` on slot ``initial_state_indices[i]`` of
    ``initial_state``; a negative slot is left as found and its output unwritten."""
    assert state is None, "the CPU GDN decode reads the pool (initial_state)"
    heads, dim_k = v.shape[2], q.shape[-1]
    scale = dim_k ** -0.5 if scale is None else scale
    qf, kf = (_at_value_heads(t[:, 0].float(), heads) for t in (q, k))
    if use_qk_l2norm:
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
    alpha = torch.exp(-torch.exp(A_log.float()) * F.softplus(a[:, 0].float() + dt_bias.float()))
    rate = torch.sigmoid(b[:, 0].float())
    out = output if output is not None else torch.zeros(v.shape, dtype=q.dtype)
    for row, slot in enumerate(initial_state_indices.tolist()):
        if slot < 0:
            continue
        y, new = _step(initial_state[slot].float(), qf[row], kf[row], v[row, 0].float(), alpha[row], rate[row], scale)
        out[row, 0] = y.to(out.dtype)
        initial_state[slot] = new.to(initial_state.dtype)
    return out, initial_state
