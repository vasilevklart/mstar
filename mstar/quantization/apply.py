"""Carrying out a ``QuantPlan`` on one rank's weights root, around its model's own weight loader.

- ``prepare`` (before the checkpoint streams in): each ``DIRECT`` layer swaps its bf16 weight for the format's
  empty storage, named as the checkpoint names it, so the model's loader (remap, stacked rules, the layer's
  own TP sharding) fills it unchanged.
- ``intercept`` (wrapping the stream): tensors of ``DEQUANT`` / ``REQUANT`` layers are paired per checkpoint
  linear (every quantized tensor the checkpoint holds for it) and come out as one bf16 ``weight``, which loads as
  a bf16 checkpoint's would; a ``DIRECT`` layer's tensors pass as they are.
- ``finalize`` (after the load): every ``DIRECT`` layer's scales are checked to have arrived; ``QUANTIZE`` /
  ``REQUANT`` layers quantize their loaded bf16 shard; the formats' kernels load before any capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from mstar.model.components.quant_linear import LinearFormat, QuantLinear, linear_format
from mstar.model.loader.base import StackedParamRule, _StackedMatcher
from mstar.quantization.checkpoint import CheckpointQuant, split_name
from mstar.quantization.formats import BF16
from mstar.quantization.resolve import Action, LayerPlan, QuantPlan

logger = logging.getLogger(__name__)

# checkpoint tensors a dequantized linear no longer needs (static activation scales)
_DROPPED = ("input_scale",)
# tensors of a checkpoint linear that load as they are whatever its format (its bias)
_PASSED = ("bias",)
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def _block_format(block: tuple[int, int] | None) -> LinearFormat:
    from mstar.model.components.fp8_formats import Fp8BlockFormat

    return Fp8BlockFormat.for_block(block)


def format_of(layer: LayerPlan) -> LinearFormat:
    """The format object a linear layer's plan runs: fp8_block at the checkpoint's block when it loads as stored."""
    if layer.action is Action.DIRECT and layer.runtime == "fp8_block":
        return _block_format(layer.block)
    return linear_format(layer.runtime)


def stored_format(stored) -> LinearFormat:
    """The format object of one checkpoint linear (a ``checkpoint.StoredBase``), to dequantize it."""
    if stored.format == "fp8_block":
        return _block_format(stored.block)
    return linear_format(stored.format)


def _is_scale(name: str, stands_for: torch.dtype | None) -> bool:
    # data tensors can hold any bit pattern; scales can't be NaN
    return "scale" in name and stands_for is not None and stands_for.is_floating_point


def _fill_unloaded(tensor: torch.Tensor, stands_for: torch.dtype) -> None:
    """Mark a scale not loaded yet: NaN, in whatever float it stands for (0xFF is NaN in e4m3 and e8m0)."""
    if stands_for.itemsize == 1:
        tensor.data.view(torch.uint8).fill_(0xFF)
    else:
        tensor.data.view(stands_for).fill_(float("nan"))


def _unloaded(tensor: torch.Tensor, stands_for: torch.dtype) -> bool:
    if stands_for.itemsize == 1:
        return bool((tensor.data.view(torch.uint8) == 0xFF).any())
    return bool(torch.isnan(tensor.data.view(stands_for)).any())


def prepare(root: nn.Module, plan: QuantPlan) -> None:
    """Give every ``DIRECT`` layer the stored format's empty storage, its scales NaN until loaded."""
    direct = plan.by_action(Action.DIRECT)
    for layer in direct:
        root.get_submodule(layer.path).quant_prepare(format_of(layer))
    mark_unloaded(root, plan)
    if direct:
        logger.info("%s: %d layers load as stored", plan.node, len(direct))


def mark_unloaded(root: nn.Module, plan: QuantPlan) -> None:
    """Mark every ``DIRECT`` layer's scales not loaded (``finalize`` refuses a layer whose marks survive the
    load). Nothing to mark on the meta device."""
    for layer in plan.by_action(Action.DIRECT):
        owner = root.get_submodule(layer.path)
        for name, dtype in owner._quant_views.items():
            if _is_scale(name, dtype) and not getattr(owner, name).is_meta:
                _fill_unloaded(getattr(owner, name), dtype)


def intercept(
    weights: Iterable[tuple[str, torch.Tensor]],
    plan: QuantPlan,
    stacked_params: list[StackedParamRule] | None = None,
    name_remapper: Callable[[str], str | None] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    ckpt: CheckpointQuant | None = None,
    prefix: str = "",
) -> Iterator[tuple[str, torch.Tensor]]:
    """``weights`` with the quantized tensors of every dequantized layer replaced by its bf16 ``weight``. A
    tensor's layer is found the way ``load_weights_into`` places it: ``name_remapper``, then the stacked rules.
    ``ckpt`` (the checkpoint's scan) says which tensors make up each checkpoint linear (its names carry
    ``prefix``, which the stream's don't); without it a linear is a ``weight`` and one fp8 scale."""
    dequant = {layer.path: layer for layer in plan.by_action(Action.DEQUANT, Action.REQUANT)}
    if not dequant:
        yield from weights
        return
    match = _StackedMatcher(list(stacked_params or []))
    located: dict[str, LayerPlan | None] = {}

    def layer_of(base: str) -> LayerPlan | None:
        if base not in located:
            mapped = name_remapper(base + ".weight") if name_remapper is not None else base + ".weight"
            path = None if mapped is None else match(mapped)[0].rpartition(".")[0]
            located[base] = dequant.get(path)
        return located[base]

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        base, suffix = split_name(name)
        layer = layer_of(base) if suffix else None
        if layer is None:
            yield name, tensor
            continue
        if suffix in _DROPPED:
            continue
        stored = ckpt.bases.get(prefix + base) if ckpt is not None else None
        if stored is not None:
            # the checkpoint's own account of this linear: every quantized tensor it holds for it
            if stored.format == BF16 or suffix in _PASSED:
                yield name, tensor
                continue
            expected = set(ckpt.suffixes[prefix + base]) - set(_DROPPED) - set(_PASSED)
            slot = pending.setdefault(base, {})
            slot[suffix] = tensor
            if set(slot) == expected:
                del pending[base]
                fmt = stored_format(stored)
                buffers = {s: fmt.from_checkpoint(s, t) for s, t in slot.items()}
                yield base + ".weight", fmt.dequantize(buffers).to(dtype)
            continue
        if suffix not in ("weight", "weight_scale_inv", "weight_scale") or (
            suffix == "weight" and tensor.dtype in _FLOATS
        ):
            # a bias, or a shard the checkpoint keeps bf16 (a mixed layer): as it is
            yield name, tensor
            continue
        slot = pending.setdefault(base, {})
        slot[suffix] = tensor
        if "weight" in slot and len(slot) == 2:
            del pending[base]
            (scale_name,) = (s for s in slot if s != "weight")
            weight, scale = slot["weight"], slot[scale_name]
            if scale_name == "weight_scale_inv":
                fmt = _pair_block_format(weight, scale, layer.block)
            else:
                fmt = linear_format("fp8_tensor" if scale.numel() == 1 else "fp8_channel")
            buffers = {"weight": fmt.from_checkpoint("weight", weight),
                       scale_name: fmt.from_checkpoint(scale_name, scale)}
            yield base + ".weight", fmt.dequantize(buffers).to(dtype)
    if pending:
        raise ValueError(
            f"{plan.node}: the checkpoint stream ended with quantized tensors missing their pair "
            f"(weight or scale): {sorted((b, sorted(s)) for b, s in pending.items())[:5]}"
        )


def _pair_block_format(weight: torch.Tensor, scale: torch.Tensor, hint: tuple[int, int] | None) -> LinearFormat:
    """The block-fp8 format of one stored pair: the layer's block, else what the shapes say (a sliced read
    hands over this rank's slice of both, which keeps the block)."""
    from mstar.quantization.checkpoint import TensorInfo, classify

    stored = classify({"weight": TensorInfo("F8_E4M3", tuple(weight.shape)),
                       "weight_scale_inv": TensorInfo("F32", tuple(scale.shape))}, "fp8", hint)
    if stored.format != "fp8_block":
        raise ValueError(f"a weight {tuple(weight.shape)} with scales {tuple(scale.shape)} is no block-fp8 pair")
    return _block_format(stored.block)


def finalize(root: nn.Module, plan: QuantPlan, device: torch.device | str | None = None) -> None:
    """Check the ``DIRECT`` layers' scales all arrived, quantize the layers loaded bf16, load the kernels."""
    formats = set()
    for layer in plan.by_action(Action.DIRECT):
        module = root.get_submodule(layer.path)
        missing = [n for n, d in module._quant_views.items()
                   if _is_scale(n, d) and _unloaded(getattr(module, n), d)]
        if missing:
            raise ValueError(f"{plan.node}: {layer.path!r} is stored {layer.stored}, but some of its scales never "
                             f"arrived from the checkpoint ({missing})")
        formats.add(module.quant_format)
    for layer in plan.by_action(Action.QUANTIZE, Action.REQUANT):
        fmt = linear_format(layer.runtime)
        formats.add(fmt)
        module = root.get_submodule(layer.path)
        if hasattr(module, "quant_quantize"):
            module.quant_quantize(fmt, tail_blocks=layer.tail)
        else:
            parent_name, _, leaf = layer.path.rpartition(".")
            setattr(root.get_submodule(parent_name), leaf, QuantLinear.from_linear(module, fmt))
    if device is not None:
        for fmt in formats:
            fmt.prepare_device(torch.device(device))
    if plan.active:
        logger.info("%s: quantization %s", plan.node, plan.summary())


__all__ = ["finalize", "format_of", "intercept", "mark_unloaded", "prepare", "stored_format"]
