"""The conductor builds the fused-MoE align op before spawning workers only for a
model that asks: every other model paid the nvcc build, or hung on its lock. It then
starts the worker processes together."""
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from mstar.conductor import conductor as conductor_mod
from mstar.conductor.conductor import Conductor


@pytest.mark.parametrize("prebuild", [False, True])
def test_only_a_model_that_asks_builds_the_op(monkeypatch, prebuild):
    built = []
    align = types.ModuleType("mstar.utils.fused_moe.align")
    align._cuda_op_available = lambda: built.append(1)
    monkeypatch.setitem(sys.modules, "mstar.utils.fused_moe.align", align)
    monkeypatch.setattr(Conductor, "shutdown", lambda self: None)  # its atexit hook
    conductor = object.__new__(Conductor)
    conductor.model = SimpleNamespace(prebuild_fused_moe=prebuild) if prebuild else object()
    conductor._sorted_ranks, conductor.worker_ids, conductor._worker_devices = [], [], []
    conductor._mps_requested, conductor._worker_processes = False, []
    conductor._launch_workers()
    assert built == ([1] if prebuild else [])


class _FakeProcess:
    """A spawned worker whose ``start`` returns only once every other worker's has begun
    (a barrier), as a spawn start waits on the child's imports; ``fails`` raises after it."""

    def __init__(self, name: str, barrier: threading.Barrier | None, fails: bool):
        self.name, self._barrier, self._fails = name, barrier, fails
        self.started_on = None

    def start(self):
        self.started_on = threading.current_thread().name
        if self._barrier is not None:
            self._barrier.wait()
        if self._fails:
            raise OSError(f"{self.name} could not spawn")


def _launch(monkeypatch, workers: int, failing: tuple[str, ...] = ()):
    """``_launch_workers`` over ``workers`` fake processes; returns the conductor and them."""
    monkeypatch.setattr(Conductor, "shutdown", lambda self: None)  # its atexit hook
    # a sequential start would wait at the barrier alone and fail it after the timeout
    barrier = threading.Barrier(workers, timeout=10) if workers > 1 else None
    made = []

    def process(target, kwargs, daemon):
        made.append(_FakeProcess(f"worker-{kwargs['worker_id']}", barrier, kwargs["worker_id"] in failing))
        return made[-1]

    monkeypatch.setattr(conductor_mod.mp, "get_context", lambda method: SimpleNamespace(Process=process))
    conductor = object.__new__(Conductor)
    conductor.model = object()
    conductor.worker_ids = [f"w{i}" for i in range(workers)]
    conductor._worker_devices = [f"cuda:{i}" for i in range(workers)]
    conductor._mps_requested, conductor._mps, conductor._worker_processes = False, None, []
    per_worker = {w: None for w in conductor.worker_ids}
    conductor._per_worker_graphs = conductor.per_worker_sharding_config = per_worker
    conductor.parallel_config = SimpleNamespace(per_worker_config=per_worker)
    unread = (
        "model_config _all_worker_graph_ids_to_graph_walks _all_worker_graph_ids_to_nodes "
        "_all_worker_graph_ids_to_dyn_loops _all_worker_graph_ids_to_instances hostname socket_path_prefix "
        "_dist_init_method enable_nvtx enable_prof log_level tensor_comm_protocol tcp_transfer_device "
        "_status_interval_s _declaration_digests"
    )
    for name in unread.split():
        setattr(conductor, name, None)
    return conductor, made


def test_workers_start_together_and_are_tracked_in_worker_order(monkeypatch):
    conductor, made = _launch(monkeypatch, 4)
    conductor._launch_workers()
    assert conductor._worker_processes == made
    assert [p.name for p in made] == ["worker-w0", "worker-w1", "worker-w2", "worker-w3"]
    assert len({p.started_on for p in made}) == 4
    assert threading.current_thread().name not in {p.started_on for p in made}


def test_a_worker_that_fails_to_start_raises_and_the_others_stay_tracked(monkeypatch):
    conductor, made = _launch(monkeypatch, 3, failing=("w1",))
    with pytest.raises(OSError, match="worker-w1 could not spawn"):
        conductor._launch_workers()
    # shutdown stops what did start
    assert conductor._worker_processes == [made[0], made[2]]


def test_one_worker_starts_on_the_conductors_thread(monkeypatch):
    conductor, made = _launch(monkeypatch, 1)
    conductor._launch_workers()
    assert conductor._worker_processes == made
    assert made[0].started_on == threading.current_thread().name
