"""The engine's side: quantization around a node's ``get_submodule``.

The engine manager loads every node through ``load_submodule``, which opens a ``node_scope`` while the model
builds and loads it. ``load_weights_into`` (every model's loader goes through it) asks ``active_scope()`` whether
the module it fills is a declared weights root, and if so the scope resolves that root's plan, prepares it and
intercepts the stream. When ``get_submodule`` returns, ``finish`` quantizes what loaded bf16. A model only declares
(``Model.get_quantization_roots``). A node with no declared root costs nothing, and a declared root with no
``quantization:`` block whose checkpoint declares no quantization loads exactly as it would without the scope.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule
from mstar.model.loader.quantization import checkpoint_quantization
from mstar.quantization import apply
from mstar.quantization.checkpoint import CheckpointQuant, scan_checkpoint
from mstar.quantization.formats import AUTO, BF16
from mstar.quantization.resolve import NO_CUDA_CAPABILITY, QUANT_METHODS, QuantizationRoot, QuantPlan, resolve_plan
from mstar.quantization.spec import NodeQuantSpec, parse_quantization

logger = logging.getLogger(__name__)

_ACTIVE: ContextVar[NodeQuantization | None] = ContextVar("mstar_quantization_scope", default=None)


def device_capability(device: torch.device | str | None) -> tuple[int, int] | None:
    """The compute capability the plan checks on ``device``: None off a GPU (nothing to check),
    ``NO_CUDA_CAPABILITY`` on a "cuda" device that isn't NVIDIA's (ROCm), where no fp8 format runs."""
    from mstar.platforms import current_platform

    if device is None or torch.device(device).type != "cuda":
        return None
    capability = current_platform.cuda_capability(torch.device(device))
    return NO_CUDA_CAPABILITY if capability is None else capability


class NodeQuantization:
    """One node's quantization while it loads: its spec, its declared roots, and each root's plan."""

    def __init__(self, node: str, spec: NodeQuantSpec, roots: list[QuantizationRoot],
                 device: torch.device | str | None = None, checkpoint: CheckpointQuant | None = None):
        """``checkpoint``: what the stream stores, when it comes from no directory (an in-memory stream)."""
        self.node = node
        self.spec = spec
        self.roots = roots
        self.device = device
        self._given = checkpoint
        self._checkpoints: dict[str, CheckpointQuant | None] = {}
        # id(root module) -> (module, plan), in load order
        self.plans: dict[int, tuple[nn.Module, QuantPlan]] = {}
        # roots that load as they would with no scope (nothing asked, nothing declared quantized)
        self._untouched: set[int] = set()
        self._finalized: set[int] = set()

    @contextlib.contextmanager
    def activated(self) -> Iterator[NodeQuantization]:
        """This scope as the active one for the body (``node_scope`` does it per node)."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def _checkpoint(self, decl: QuantizationRoot) -> CheckpointQuant | None:
        if self._given is not None:
            return self._given
        directory = decl.checkpoint_dir()
        if directory is None:
            return None
        if directory not in self._checkpoints:
            self._checkpoints[directory] = scan_checkpoint(directory)
        return self._checkpoints[directory]

    def _declares_quantization(self, decl: QuantizationRoot) -> bool:
        if self._given is not None:
            return self._given.quantized
        directory = decl.checkpoint_dir()
        return directory is not None and checkpoint_quantization(directory) is not None

    def plan_for(self, module: nn.Module) -> QuantPlan | None:
        """The plan for ``module`` if it is one of this node's declared roots (resolved, checked and prepared
        on first sight), else None. None too when the spec asks nothing and the checkpoint declares no
        quantization: the root then loads as it would with no scope."""
        key = id(module)
        if key in self.plans:
            return self.plans[key][1]
        if key in self._untouched:
            return None
        decl = next((r for r in self.roots if type(module) is r.root_type), None)
        if decl is None:
            return None
        if self.spec.is_default and not self._declares_quantization(decl):
            self._untouched.add(key)
            return None
        plan = resolve_plan(module, decl, self.spec, self._checkpoint(decl), device_capability(self.device))
        if plan.problems:
            raise ValueError(f"quantization of node {self.node!r} refused:\n  - " + "\n  - ".join(plan.problems))
        for note in plan.notes:
            logger.info("%s: %s", self.node, note)
        apply.prepare(module, plan)
        self.plans[key] = (module, plan)
        return plan

    def wrap_load(
        self, module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params: list[StackedParamRule] | None, name_remapper: Callable[[str], str | None] | None,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        plan = self.plan_for(module)
        if plan is None or not plan.active:
            return weights
        decl = next(r for r in self.roots if type(module) is r.root_type)
        return apply.intercept(weights, plan, stacked_params, name_remapper, ckpt=self._checkpoint(decl),
                               prefix=decl.rules.prefix)

    def finish(self, submodule: nn.Module | None) -> None:
        """Finalize every root this node loaded; refuse a spec that asked for a format no root took."""
        for key, (module, plan) in self.plans.items():
            if key not in self._finalized:
                apply.finalize(module, plan, self.device)
                self._finalized.add(key)
        asked = self.spec.format not in (AUTO, BF16) or any(rule.format != BF16 for rule in self.spec.rules)
        if submodule is not None and asked and not self.plans:
            raise ValueError(
                f"quantization asks {self.spec.format} for node {self.node!r}, but its weights never went through "
                f"a declared root ({[r.root_type.__name__ for r in self.roots]})"
            )


def active_scope() -> NodeQuantization | None:
    return _ACTIVE.get()


def quant_methods() -> tuple[str, ...]:
    """The checkpoint ``quant_method``s a declared root's loader lets through the load guard
    (``iter_safetensors_shards(quant_methods=...)``): the plan's while a node scope is active, none otherwise, so a
    load outside the engine's scope refuses a quantized checkpoint as before."""
    return QUANT_METHODS if active_scope() is not None else ()


@contextlib.contextmanager
def node_scope(
    model: Any, node: str, model_config: Mapping[str, Any], device: torch.device | str | None = None,
) -> Iterator[NodeQuantization | None]:
    """Quantization for ``node`` while the body builds and loads it; yields None when the model declares no root
    there (``validate_deployment`` refused a block that asks something of such a node)."""
    roots = [r for r in getattr(model, "get_quantization_roots", list)() if r.node == node]
    if not roots:
        yield None
        return
    spec = parse_quantization(model_config.get("quantization")).for_node(node)
    scope = NodeQuantization(node, spec, roots, device)
    with scope.activated():
        yield scope


def load_submodule(model: Any, node: str, model_config: Mapping[str, Any] | None = None,
                   device: torch.device | str = "cpu", **kwargs: Any):
    """``model.get_submodule(node, device, **kwargs)`` under the node's quantization scope: how the engine
    manager loads a node, for anything else that loads one (tests, tools)."""
    with node_scope(model, node, model_config or {}, device) as scope:
        submodule = model.get_submodule(node, device, **kwargs)
        if scope is not None:
            scope.finish(submodule)
    return submodule
