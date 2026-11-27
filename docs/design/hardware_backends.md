# Hardware backends

How M\* decides what runs on which hardware, what a platform has to say about itself, and how a platform that is not in the tree plugs in. This note describes the contract in `mstar/platforms` and the pieces of the engine that read it. Parts marked **proposed** are directions, not code.

## 1. Scope and status

A platform is **supported** when it is on `main`, installable from the docs, and tested on that hardware.

| Platform | `MSTAR_PLATFORM` | Status |
|---|---|---|
| NVIDIA GPUs (CUDA) | `cuda` | Supported |
| AMD GPUs (ROCm) | `rocm` | To be set by the maintainers at launch |
| Intel GPUs (XPU) | `xpu` | Experimental, one model |
| No accelerator | `cpu` | Tests and tooling |

The platform contract is versioned (`PLATFORM_API_VERSION` in `mstar/platforms/registry.py`) and may change before 1.0.

## 2. How M\* picks an implementation today

Two invariants from `AGENTS.md` set the shape. Models declare and the engine executes (invariant 1), so a model names the resources it needs and the resource chooses its kernels. Code that branches on hardware asks `mstar.platforms.current_platform` (invariant 8) rather than reading `torch.version.hip`, the device type or a raw compute capability.

The pieces that already exist:

- `Platform` (`mstar/platforms/interface.py`): which stack this is, the GPU architecture, environment defaults applied before the first device call, Inductor settings for a compile.
- `DeviceCapability` (`mstar/utils/device.py`): whether a device is an accelerator, can pin memory, and has a graph backend.
- `AcceleratorGraphBackend` (`mstar/engine/accelerator_graph_backend.py`): capture and replay per device type, with a CUDA and an XPU implementation.
- The kernel registry (`mstar/kernels/registry.py`): ops with implementations that say why they can't run a target (`Impl.check`), and tuned tables per GPU.
- `AttnBackend.AUTO` (`mstar/engine/resources/attn/config.py`): the platform's default attention, decided when the resource is built.
- Custom ops registered per torch device type (`torch.library`, `register_kernel("cuda")`), and `PlatformDispatch` (`mstar/platforms/dispatch.py`) for a callable with one implementation per platform.

A device type is not enough to choose a kernel. PyTorch's ROCm build calls AMD GPUs `cuda` devices and reports an MI300X as compute capability 9.4, so neither the device type nor the capability tells you the vendor. A kernel registered for the `"cuda"` dispatch key runs on both.

## 3. Three kinds of hardware

| Kind | What the engine can assume | What it can use |
|---|---|---|
| Stream-and-graph GPUs | A torch device with streams and events, graph capture and replay, eager fallback for any shape, Triton | All of the engine. NVIDIA, AMD and Intel GPUs are this kind |
| Compile-to-executable accelerators | A torch device and a `torch.compile` backend; one executable per static shape; no stream capture, and often no Triton | Eager execution of torch references. Graph-mode serving needs the proposed backend kind in section 9, a compile-backend hook and an engine that imports without Triton |
| Remote endpoints | No device: a network API that takes tokens or text and streams them back | Not a platform. A graph node of its own (section 10, proposed) |

## 4. The platform contract

A `Platform` says what the hardware is and how to start a process on it:

- **Identity:** `name`, `is_cuda()` and friends for error text and for what really is about the vendor, `arch()`.
- **Capabilities:** what a device can do (section 5).
- **Process start:** `process_env_defaults()`, applied before the first device call because runtimes read their environment at init; PCI addresses of the visible devices. Today the defaults applied are those of the platform torch was built for, so an out-of-tree platform's are not applied yet.
- **Compilation:** `inductor_overrides()` for codegen settings on that stack.
- **Collectives:** today the process-group backend is torch's default for the device type (`mstar/distributed/communication.py`).

What a platform never does: import a kernel, choose an engine policy, or supply a worker or a scheduler. Which kernel runs stays the choice of the resource that runs it.

Rules that keep it small:

1. **Facts and names, not kernels.** A platform answers questions; it does not hold implementations.
2. **Ask a capability, not a vendor.** `not current_platform.is_rocm()` reads as "NVIDIA" while there are two vendors and silently puts a third on NVIDIA's path. `capabilities().nvidia_triton_extras` says what the code needs.
3. **Decide once, when a resource is built, and log it.** A slot is never resolved inside a forward that is compiled or captured. A capability may be asked there: Dynamo folds the answer into the graph.
4. **Refuse with a reason.** A path with nothing that runs on this platform is a startup error naming the platform and what is missing, not NVIDIA's implementation by default.
5. **A small hook budget.** A hook is added when two packages need the same answer. A question only one resource asks is that resource's `check`.

**Versioning.** A platform's `PlatformEntry` carries the `PLATFORM_API_VERSION` it was written for. The registry refuses an entry for another version, with both numbers in the message.

## 5. Capabilities

`mstar.platforms.capabilities.capabilities(device=None, platform=None)` returns an immutable `Capabilities`. Every default is the conservative answer, so a CPU device, and a platform M\* knows nothing about, get the safe path.

| Field | Meaning | Default | NVIDIA | AMD | Intel | Read by |
|---|---|---|---|---|---|---|
| `nvidia_triton_extras` | libdevice, inline PTX, `triton.language.extra.cuda` | `False` | `True` | `False` | `False` | GLM-5-next's fused decode kernels |
| `pdl` | programmatic dependent launch | `False` | from sm_90 | `False` | `False` | the fused MoE decode chain |
| `pinned_host_memory` | host tensors can be pinned for async copies to the device | `False` | `True` | `True` | `True` | the block-causal attention planner |

A field is added with the first code that reads it, so the table is short. Facts that code already asks elsewhere stay where they are: `Platform.arch()`, `has_cuda_capability()` and `has_block_scaled_gemm()` on the platform, and `DeviceCapability` for what torch's runtime for a device type offers (streams, a graph backend) without touching the device.

How a question is answered: a CPU device gets the defaults on every platform. Otherwise the platform's own `capabilities(device)` method answers, then a builder registered for its name (`register_capabilities`), which is how the in-tree platforms answer; a platform with neither gets the defaults.

Answers are cached per platform instance and device. Answering for an NVIDIA GPU reads its compute capability, which initializes the runtime, so this is for worker-side code once the device is chosen. Under `torch.compile`, Dynamo calls `capabilities` at trace time and folds the answer into the graph.

## 6. Strategy slots

A **strategy slot** (`mstar/platforms/strategy.py`) holds the implementations of one resource family or op, owned by that resource, in the order it prefers them:

```python
PAGE_COPIER = StrategySlot("kv.page_copier", reference="torch")
PAGE_COPIER.register("libcudart", lambda: CudartCopier(), platforms={"cuda"}, verify=_verify)
PAGE_COPIER.register("cuda.bindings", lambda: CudaBindingsCopier(), platforms={"cuda"}, verify=_verify)
PAGE_COPIER.register("libamdhip64", lambda: HipCopier(), platforms={"rocm"}, verify=_verify)
PAGE_COPIER.register("torch", lambda: TorchCopier())

copier = PAGE_COPIER.resolve().impl
```

An implementation declares the platforms it is for (`"rocm"`, or `"rocm:gfx950"` for one architecture; none for every platform), a `check(target)` that returns why it can't run a target or `None`, and optionally a `verify(impl)` that tests a built instance. Its factory does its own imports, so registering it loads nothing.

`resolve(requested=None, target=None)` decides afresh on each call:

1. A **requested** name runs, or raises `StrategyUnavailableError` with the slot, the platform and the reason. A named choice never falls back.
2. Otherwise the first implementation registered for this platform whose check passes, whose factory builds and whose verify passes. Registration order is the only ranking.
3. Otherwise the slot's **reference**.
4. Otherwise `StrategyUnavailableError`, naming the platform and every implementation passed over with its reason.

Each one passed over is kept in `Resolved.skipped` with its reason, so the resource can log why it got what it got. A factory that raises `ImportError`, `OSError`, `AttributeError` or `RuntimeError` counts as unavailable; any other exception is a bug and propagates.

**A slot should have a torch reference.** It defines the semantics, it is what a new platform runs before anyone writes a kernel for it, and it is what tuning sweeps compare candidates against (`Tolerance` in `mstar/kernels/registry.py`). A slot built without one refuses on a platform it has no implementation for (step 4).

Proposed:

- A platform names its default for a slot, tried after a requested name and before the chain.
- An implementation also says which steps it can't capture and how far its output may sit from the reference.
- The choice joins the resource's fingerprint. The ranks of a lockstep instance must make the same choice (invariant 11), so its inputs (the platform, the config, the installed libraries) must agree across ranks, and a mismatch should fail at startup.

Slots today, and what each picks by default:

| Slot | `cuda` | `rocm` | `xpu`, `cpu`, any other |
|---|---|---|---|
| `kv.page_copier` | `libcudart`, then `cuda.bindings`, then `torch` | `libamdhip64`, then `torch` | `torch` |

Once there is more than one slot, this table should be generated from the registry.

## 7. Configuration

- **A non-default implementation** is named in the serving YAML, as attention backends are today (`resources: {attn: {backend: xpu_paged}}` in `configs/bagel_xpu_cfg_tp2.yaml`); left out, `AUTO` is the platform's default. No strategy slot reads the YAML yet; `resolve(requested=...)` is where a configured name goes.
- **Layout differences per platform** (proposed): `platform_overrides:` in the serving config.
- **Environment variables** are for what has to be known before the first device call: `MSTAR_PLATFORM` (which platform) and `MSTAR_KERNEL_TABLES` (tuned tables). Both are documented in `docs/environment_variables.rst`.
- **Not configuration:** capability facts, which come from the hardware, and per-op switches for ops that are normally fused.

## 8. Out-of-tree platforms

A package outside the tree adds a platform through the `mstar.platforms` entry-point group, declared in its own `pyproject.toml`. M\*'s packaging needs no change. The entry point's name is the platform's name, and it names a callable that returns a `PlatformEntry`, or `None` when the package can't serve this host.

Selection (`mstar.platforms.registry.select_platform`, which `resolve_platform` asks):

1. `MSTAR_PLATFORM` names the platform. Only that plugin is loaded, so no other vendor's SDK is imported. An unknown name is an error that lists the known ones, and so is a platform this torch is not built for (`rocm` on a CUDA build).
2. Unset: the in-tree accelerators, in the order M\* always used (CUDA or ROCm, told apart by `torch.version.hip`, then XPU).
3. Then the installed plugins whose devices are available. Exactly one wins; more than one is an error that asks for `MSTAR_PLATFORM`.
4. Then `cpu`.

In-tree accelerators come first so that an installed plugin cannot take over a GPU host by accident; `MSTAR_PLATFORM` is the explicit override. In a scan, a plugin that fails to load is logged and skipped; a plugin named by `MSTAR_PLATFORM` that fails raises.

A plugin may supply a platform and its capabilities, implementations registered into slots, and a graph backend (`register_graph_backend`). It may not supply a worker, a scheduler, a resource lifecycle or replacements for model code.

Vendors whose code is in the tree track `main` by construction. A plugin pins the contract version it was written for.

```toml
# the plugin's pyproject.toml
[project.entry-points."mstar.platforms"]
myaccel = "mstar_myaccel:platform_entry"
```

```python
# mstar_myaccel/__init__.py
import torch

from mstar.platforms import Platform
from mstar.platforms.capabilities import Capabilities
from mstar.platforms.registry import PlatformEntry


class MyAccelPlatform(Platform):
    name = "myaccel"

    def capabilities(self, device=None):
        return Capabilities(pinned_host_memory=True)


def platform_entry():
    if getattr(torch, "myaccel", None) is None:  # this torch has no such device
        return None
    return PlatformEntry("myaccel", MyAccelPlatform, device_type="myaccel",
                         built_for=lambda: True, available=lambda: torch.myaccel.is_available())
```

An entry's `factory` should be a plain constructor. `torch.compile` traces it whenever a compiled forward asks `current_platform`, as it traces the in-tree platforms' constructors; the selection itself is not traced.

Its tests can run on a CPU with `mstar.platforms.testing`:

```python
from mstar.engine.resources.kv.page_copier import PAGE_COPIER
from mstar.platforms.testing import FakePlatform, use_platform


def test_an_unknown_platform_copies_pages_with_the_reference():
    with use_platform(FakePlatform("myaccel")):
        assert PAGE_COPIER.resolve().name == "torch"
```

A second entry-point group, for modules that register strategies when imported, is a follow-up.

## 9. Graph backends

An `AcceleratorGraphBackend` gives the graph runners everything they need from a device's runtime: availability, streams and stream contexts, graph creation, a memory pool, capture, recovery from a failed capture, and allocated memory. A device type with no backend runs every step eagerly; the runner logs that it skipped capture.

`register_graph_backend(device_type, backend_cls)` adds one for a new device type. `DeviceCapability` keeps the backend it found per device, so a backend must be registered before the first lookup of a device of that type.

**Proposed:** a second kind of backend for accelerators with no stream capture, whose "capture" compiles a bucket into one executable and whose replay runs it. M\*'s serving shape already fits (`dynamic=False`, one capture per bucket, pad and replay). What does not fit yet is the eager fallback for shapes outside the buckets and eager regions inside a captured forward (`torch.compiler.disable` and opaque custom ops), which have no place in one compiled executable.

## 10. Remote-endpoint nodes (proposed)

For hardware that is reachable only through a network API. A graph node hosted by a worker with no accelerator sends its input to the endpoint and streams the output into the graph. Constraints:

- Its edges carry tokens or text.
- The endpoint holds the request's state, so the node can't rerun a request once output has started.
- The endpoint's rate limits act as the node's admission budget.
- It is chosen in config, per node.
- Requests leave the deployment, so it is opt-in.

## 11. Adding a platform: a checklist

1. A `Platform` subclass and its capabilities; in the tree, a row in the registry's detection order.
2. A graph backend, or eager execution.
3. Implementations for the slots on the target models' step path, or confirmation that their torch references are fast enough.
4. A collectives backend for the device type.
5. Packaging: wheels, extras and the torch versions that work.
6. Hardware tests on that platform.
7. A row in the status table above.

## 12. Testing

- **The contract test** (`test/modular/test_platform_contract.py`) runs a `FakePlatform` through each slot and gate, and each must reach a reference or a refusal that names the platform. A path where an unknown platform still gets an in-tree one's treatment is written as a strict expected failure naming its fix, so the list of them is the porting checklist; today it holds `AUTO` attention and the process environment defaults.
- **Conformance** (proposed): each implementation against its slot's reference, at its tolerance, on each platform it registers for.
- **Hardware tests** run per platform. CI has no accelerator today, so a change that runs on a device reports which hardware it was run on.

## 13. Non-goals and open questions

- One model definition per vendor is not a goal.
- A stable plugin API before 1.0 is not promised.
- Open: should the in-tree detectors win over an installed plugin, as here, or lose to it, as in some other engines?
- Open: which torch versions each platform supports, given one torch cap for all of them.

## 14. Prior art

- vLLM's plugin system: https://docs.vllm.ai/en/latest/design/plugin_system/
- SGLang's platform and kernel registries: https://github.com/sgl-project/sglang
- PyTorch's accelerator integration guide: https://docs.pytorch.org/docs/stable/accelerator/index.html
- PyTorch's Accelerator Integration Working Group, H1 2026 update: https://pytorch.org/blog/pytorch-hardware-enablement-updates-from-the-acceleration-integration-working-group/
