"""Compile-cache bundles (mstar/utils/compile_bundle.py, ``mstar cache``): packing
appends only what is new, unpacking restores the caches and marks them (keeping
what a cache has, merging MIOpen's databases, one unpack per cache dir at a
time), a bundle for another stack is refused, and a bundle root laid out like the
golden copies the AMD benchmark scripts seed from (uncompressed per-kind tars
of ``./`` members, no manifest) reads as a bundle."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from mstar.cli import cache as cache_cli
from mstar.cli.main import build_parser
from mstar.utils import compile_bundle as cb

KEY = cb.StackKey(product="mi325x-vf", arch="gfx942", torch="2.12.1+rocm7.2", runtime="hip-7.2.53211",
                  triton="3.7.1")


def _write(path: Path, data: str, mtime: int = 1_700_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    os.utime(path, (mtime, mtime))


def _caches(root: Path) -> dict[str, Path]:
    """M*'s default layout: Triton's dir inside Inductor's; MIOpen beside them."""
    dirs = {"inductor": root / "inductor", "triton": root / "inductor" / "triton", "miopen": root / "miopen"}
    _write(dirs["inductor"] / "fxgraph" / "ab" / "entry", "graph")
    _write(dirs["inductor"] / "wq" / "kernel.py", "code")
    _write(dirs["triton"] / "HASH" / "k.hsaco", "binary")
    _write(dirs["miopen"] / "gfx942130.ukdb", "kernels")
    _write(dirs["miopen"] / "gfx942130.HIP.ufdb.txt", "conv-a=Solver1:0.5\n")
    return dirs


def _names(tar: Path) -> list[str]:
    with tarfile.open(tar) as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def _tree(root: Path) -> dict[str, str]:
    """The files under ``root``, less the lock unpack leaves in each cache dir."""
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != cb.DEST_LOCK}


def test_key_path_round_trips_and_names_what_differs():
    assert KEY.path == "mi325x-vf/torch-2.12.1+rocm7.2-hip-7.2.53211-triton-3.7.1"
    parsed = cb.StackKey.from_path(*KEY.path.split("/"))
    assert parsed == cb.StackKey(**{**KEY.__dict__, "arch": None})
    assert KEY.differences(parsed) == []  # a manifest-less bundle doesn't know its arch
    other = cb.StackKey("mi300x", "gfx942", KEY.torch, KEY.runtime, "3.6.0")
    assert KEY.differences(other) == ["triton 3.6.0 (here 3.7.1)"]  # the product alone is no reason
    assert cb.StackKey.from_path("x", "not-a-stack") is None


def test_pack_then_unpack_restores_the_caches(tmp_path):
    dirs = _caches(tmp_path / "built")
    bundle, results = cb.pack(tmp_path / "bundles", KEY, dirs, note="warm-up", log=lambda _: None)
    assert bundle == tmp_path / "bundles" / KEY.path
    assert {r.kind: r.added for r in results} == {"inductor": 2, "triton": 1, "miopen": 2}
    # Triton's dir inside Inductor's goes in its own tar only
    assert sorted(_names(bundle / "inductor.tar")) == ["./fxgraph/ab/entry", "./wq/kernel.py"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["key"]["arch"] == "gfx942" and manifest["history"][0]["note"] == "warm-up"

    fresh = {"inductor": tmp_path / "new" / "i", "triton": tmp_path / "new" / "t", "miopen": tmp_path / "new" / "m"}
    done = cb.unpack(tmp_path / "bundles", KEY, fresh, log=lambda _: None)
    assert done.bundle == bundle and set(done.kinds) == {"inductor", "triton", "miopen"}
    assert _tree(fresh["triton"]) == _tree(dirs["triton"])
    assert _tree(fresh["miopen"]) == _tree(dirs["miopen"])
    restored = _tree(fresh["inductor"])
    marker = json.loads(restored.pop(cb.MARKER))
    assert marker["bundle"] == str(bundle) and marker["key"]["triton"] == "3.7.1"
    assert restored == {k: v for k, v in _tree(dirs["inductor"]).items() if not k.startswith("triton/")}
    assert os.stat(fresh["triton"] / "HASH" / "k.hsaco").st_mtime == 1_700_000_000


def test_packing_inductor_alone_still_leaves_its_triton_dir_out(tmp_path):
    dirs = _caches(tmp_path / "built")
    bundle, results = cb.pack(tmp_path / "b", KEY, dirs, kinds=["inductor"], log=lambda _: None)
    assert [r.kind for r in results] == ["inductor"]
    assert sorted(_names(bundle / "inductor.tar")) == ["./fxgraph/ab/entry", "./wq/kernel.py"]


def test_a_second_pack_appends_only_new_and_changed_files(tmp_path):
    dirs = _caches(tmp_path / "built")
    root = tmp_path / "bundles"
    bundle, _ = cb.pack(root, KEY, dirs, log=lambda _: None)
    _, again = cb.pack(root, KEY, dirs, log=lambda _: None)
    assert all(r.added == 0 for r in again)
    _write(dirs["inductor"] / "fxgraph" / "cd" / "entry", "another graph")
    _write(dirs["miopen"] / "gfx942130.HIP.ufdb.txt", "conv-a=Solver2:0.4\n", mtime=1_700_000_100)
    _, third = cb.pack(root, KEY, dirs, log=lambda _: None)
    assert {r.kind: (r.added, r.members) for r in third} == {"inductor": (1, 3), "triton": (0, 1), "miopen": (1, 2)}
    assert len(_names(bundle / "miopen.tar")) == 3  # appended, not rewritten
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t", "miopen": tmp_path / "o" / "m"}
    cb.unpack(bundle, KEY, out, log=lambda _: None)
    assert (out["miopen"] / "gfx942130.HIP.ufdb.txt").read_text() == "conv-a=Solver2:0.4\n"  # this host's record
    assert len(json.loads((bundle / "manifest.json").read_text())["history"]) == 3


def _kernel_db(path: Path, rows: list[tuple[str, str]]) -> None:
    """MIOpen's kernel DB schema: a row id and a unique (name, args)."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS `kern_db` (`id` INTEGER PRIMARY KEY ASC, `kernel_name` TEXT NOT NULL, "
                "`kernel_args` TEXT NOT NULL, `kernel_blob` BLOB NOT NULL)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS `idx_kern_db` ON kern_db(kernel_name, kernel_args)")
    con.executemany("INSERT INTO kern_db (kernel_name, kernel_args, kernel_blob) VALUES (?, ?, ?)",
                    [(n, a, b"bin") for n, a in rows])
    con.commit()
    con.close()


def test_miopen_databases_from_separate_warm_ups_are_merged(tmp_path):
    """Two hosts warmed different models from empty caches: MIOpen's find DB and
    kernel DB are one file each, so packing the second must keep the first's
    records rather than replace the file."""
    import sqlite3

    first, second = tmp_path / "a" / "miopen", tmp_path / "b" / "miopen"
    _write(first / "x.ufdb.txt", "conv-a=S1:0.5\nconv-b=S1:0.7\n")
    _kernel_db(first / "x.ukdb", [("k1", "-O3"), ("k2", "-O3")])
    _write(second / "x.ufdb.txt", "conv-b=S2:0.6\nconv-c=S1:0.2\n", mtime=1_700_000_100)
    _kernel_db(second / "x.ukdb", [("k2", "-O3"), ("k3", "-O3")])
    cb.pack(tmp_path / "bundles", KEY, {"miopen": first}, log=lambda _: None)
    _, res = cb.pack(tmp_path / "bundles", KEY, {"miopen": second}, log=lambda _: None)
    assert res[0].added == 2
    out = {"inductor": tmp_path / "o" / "i", "miopen": tmp_path / "o" / "m"}
    cb.unpack(tmp_path / "bundles", KEY, out, log=lambda _: None)
    assert (out["miopen"] / "x.ufdb.txt").read_text() == "conv-a=S1:0.5\nconv-b=S2:0.6\nconv-c=S1:0.2\n"
    con = sqlite3.connect(out["miopen"] / "x.ukdb")
    assert sorted(r[0] for r in con.execute("SELECT kernel_name FROM kern_db")) == ["k1", "k2", "k3"]
    con.close()
    # packing the merged result again adds nothing
    _, res = cb.pack(tmp_path / "bundles", KEY, {"miopen": out["miopen"]}, log=lambda _: None)
    assert res[0].added == 0


def test_a_bundle_for_another_stack_is_refused_unless_forced(tmp_path):
    dirs = _caches(tmp_path / "built")
    other = cb.StackKey(KEY.product, KEY.arch, "2.11.0+rocm7.1", KEY.runtime, KEY.triton)
    bundle, _ = cb.pack(tmp_path / "bundles", other, dirs, log=lambda _: None)
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t"}
    with pytest.raises(cb.BundleMismatchError, match="has no bundle for"):
        cb.unpack(tmp_path / "bundles", KEY, out, log=lambda _: None)
    with pytest.raises(cb.BundleMismatchError, match="torch 2.11.0"):
        cb.unpack(bundle, KEY, out, log=lambda _: None)
    cb.unpack(bundle, KEY, out, force=True, log=lambda _: None)
    ok, line = cb.startup_line(KEY, out)
    assert not ok and "torch 2.11.0+rocm7.1 (here 2.12.1+rocm7.2)" in line


CUDA_KEY = cb.StackKey(product="nvidia-h200", arch="sm_90", torch="2.12.1+cu129", runtime="cuda-12.9",
                       triton="3.7.1")


def test_on_cuda_a_bundle_from_another_gpu_model_is_refused_unless_forced(tmp_path):
    """Inductor keys CUDA graphs on the GPU's name, so an H100's are never read
    on an H200; on ROCm (above) the product alone is no reason."""
    h100 = cb.StackKey("nvidia-h100-80gb-hbm3", CUDA_KEY.arch, CUDA_KEY.torch, CUDA_KEY.runtime, CUDA_KEY.triton)
    assert CUDA_KEY.differences(h100) == ["product nvidia-h100-80gb-hbm3 (here nvidia-h200)"]
    dirs = _caches(tmp_path / "built")
    bundle, _ = cb.pack(tmp_path / "bundles", h100, {k: dirs[k] for k in ("inductor", "triton")}, log=lambda _: None)
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t"}
    with pytest.raises(cb.BundleMismatchError, match="has no bundle for nvidia-h200/"):
        cb.unpack(tmp_path / "bundles", CUDA_KEY, out, log=lambda _: None)
    with pytest.raises(cb.BundleMismatchError, match=r"product nvidia-h100-80gb-hbm3 \(here nvidia-h200\)"):
        cb.unpack(bundle, CUDA_KEY, out, log=lambda _: None)
    assert not out["inductor"].exists()
    cb.unpack(bundle, CUDA_KEY, out, force=True, log=lambda _: None)
    ok, line = cb.startup_line(CUDA_KEY, out)
    assert not ok and "its entries are not used" in line


def test_a_golden_copy_layout_reads_as_a_bundle(tmp_path):
    """$WORK/.cache/compile-golden/<product>/<stack>/{inductor,triton,miopen}.tar,
    made with ``tar -cf X.tar -C dir .``: no manifest, a ``./`` directory member."""
    built = _caches(tmp_path / "built")
    golden = tmp_path / "golden" / KEY.path
    golden.mkdir(parents=True)
    for kind in ("inductor", "triton", "miopen"):
        exclude = ["--exclude=./triton"] if kind == "inductor" else []
        subprocess.run(["tar", "-cf", str(golden / f"{kind}.tar"), *exclude, "-C", str(built[kind]), "."],
                       check=True, env={**os.environ, "COPYFILE_DISABLE": "1"})
    assert cb.find_bundles(tmp_path / "golden") == [golden]
    assert cb.bundle_key(golden) == cb.StackKey(**{**KEY.__dict__, "arch": None})
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t", "miopen": tmp_path / "o" / "m"}
    cb.unpack(tmp_path / "golden", KEY, out, log=lambda _: None)
    assert _tree(out["miopen"]) == _tree(built["miopen"])
    # packing into it appends to the same tars and adds a manifest
    _write(built["triton"] / "NEW" / "k.hsaco", "binary 2")
    _, res = cb.pack(tmp_path / "golden", KEY, built, log=lambda _: None)
    assert {r.kind: r.added for r in res} == {"inductor": 0, "triton": 1, "miopen": 0}
    assert (golden / "manifest.json").is_file()


def test_startup_line_says_whether_the_unpacked_bundle_matches(tmp_path):
    dirs = _caches(tmp_path / "built")
    ok, line = cb.startup_line(KEY, dirs)
    assert ok and "no bundle unpacked" in line and KEY.path in line
    cb.pack(tmp_path / "b", KEY, dirs, log=lambda _: None)
    cb.unpack(tmp_path / "b", KEY, dirs, log=lambda _: None)
    ok, line = cb.startup_line(KEY, dirs)
    assert ok and line.endswith("matches")
    elsewhere = cb.StackKey("mi300x", KEY.arch, KEY.torch, KEY.runtime, KEY.triton)
    ok, line = cb.startup_line(elsewhere, dirs)
    assert ok and line.endswith("tuned on mi325x-vf")


def test_miopen_kernels_without_a_dir_of_their_own_go_to_the_db_dir(tmp_path):
    built = {"inductor": tmp_path / "i", "miopen": tmp_path / "db", "miopen-kernels": tmp_path / "kc"}
    _write(built["miopen"] / "x.ufdb.txt", "db")
    _write(built["miopen-kernels"] / "3.5.1" / "x.ukdb", "kernels")
    bundle, _ = cb.pack(tmp_path / "b", KEY, built, log=lambda _: None)
    assert _names(bundle / "miopen-kernels.tar") == ["./3.5.1/x.ukdb"]
    shared = {"inductor": tmp_path / "o" / "i", "miopen": tmp_path / "o" / "m"}
    cb.unpack(bundle, KEY, shared, log=lambda _: None)
    assert (shared["miopen"] / "3.5.1" / "x.ukdb").read_text() == "kernels"


def test_unpack_merges_miopen_databases_with_the_hosts(tmp_path):
    """MIOpen keeps each database in one file, so unpacking into a host that has
    one must add the bundle's records to it (the bundle's winning on the same
    key), not replace it."""
    import sqlite3

    built = tmp_path / "built" / "miopen"
    _write(built / "x.ufdb.txt", "conv-a=S1:0.5\n")
    _kernel_db(built / "x.ukdb", [("k1", "-O3")])
    cb.pack(tmp_path / "b", KEY, {"miopen": built}, log=lambda _: None)
    host = {"inductor": tmp_path / "h" / "i", "miopen": tmp_path / "h" / "m"}
    _write(host["miopen"] / "x.ufdb.txt", "conv-a=S9:0.9\nconv-z=S9:0.1\n")
    _kernel_db(host["miopen"] / "x.ukdb", [("k9", "-O3")])
    lines = []
    cb.unpack(tmp_path / "b", KEY, host, log=lines.append)
    assert "miopen: 0 files added" in lines[0] and "2 merged" in lines[0]
    assert (host["miopen"] / "x.ufdb.txt").read_text() == "conv-a=S1:0.5\nconv-z=S9:0.1\n"
    con = sqlite3.connect(host["miopen"] / "x.ukdb")
    assert sorted(r[0] for r in con.execute("SELECT kernel_name FROM kern_db")) == ["k1", "k9"]
    con.close()

    # the host now has every record the bundle has, so a second unpack leaves its files alone
    def files():
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in host["miopen"].glob("x.*")}

    before = files()
    cb.unpack(tmp_path / "b", KEY, host, log=lines.append)
    assert "0 merged, 2 already there" in lines[-1]
    assert files() == before


def test_unpack_keeps_the_entries_a_cache_already_has(tmp_path):
    dirs = _caches(tmp_path / "built")
    root = tmp_path / "b"
    cb.pack(root, KEY, dirs, log=lambda _: None)
    _write(dirs["inductor"] / "wq" / "kernel.py", "code 2", mtime=1_700_000_100)
    cb.pack(root, KEY, dirs, log=lambda _: None)  # inductor.tar now holds two copies of it
    host = {"inductor": tmp_path / "h" / "i", "triton": tmp_path / "h" / "t"}
    _write(host["inductor"] / "fxgraph" / "ab" / "entry", "this host's graph")
    cb.unpack(root, KEY, host, log=lambda _: None)
    assert (host["inductor"] / "fxgraph" / "ab" / "entry").read_text() == "this host's graph"
    assert (host["inductor"] / "wq" / "kernel.py").read_text() == "code 2"  # the tar's last copy
    # the marker and the lock stay out of a pack of what was unpacked, and mtimes survive
    fresh = {"inductor": tmp_path / "f" / "i", "triton": tmp_path / "f" / "t", "miopen": tmp_path / "f" / "m"}
    cb.unpack(root, KEY, fresh, log=lambda _: None)
    assert (fresh["inductor"] / cb.DEST_LOCK).is_file() and (fresh["inductor"] / cb.MARKER).is_file()
    _, again = cb.pack(root, KEY, fresh, log=lambda _: None)
    assert {r.kind: r.added for r in again} == {"inductor": 0, "triton": 0, "miopen": 0}


def test_unpack_waits_for_the_lock_on_its_destination(tmp_path):
    """Two servers starting on one host unpack into the same caches one at a time."""
    import threading

    dirs = _caches(tmp_path / "built")
    cb.pack(tmp_path / "b", KEY, dirs, log=lambda _: None)
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t"}
    out["inductor"].mkdir(parents=True)
    done = threading.Event()

    def unpack():
        cb.unpack(tmp_path / "b", KEY, out, log=lambda _: None)
        done.set()

    with cb._locked(out["inductor"] / cb.DEST_LOCK):
        t = threading.Thread(target=unpack, daemon=True)
        t.start()
        assert not done.wait(0.5)
        assert not (out["inductor"] / "fxgraph").exists()
    t.join(10)
    assert done.is_set() and (out["inductor"] / "fxgraph" / "ab" / "entry").read_text() == "graph"


def test_unpack_refuses_a_member_outside_the_cache_dir(tmp_path):
    bundle = tmp_path / "b" / KEY.path
    bundle.mkdir(parents=True)
    _write(tmp_path / "payload", "x")
    with tarfile.open(bundle / "triton.tar", "w") as tf:
        tf.add(tmp_path / "payload", arcname="../escaped")
    out = {"inductor": tmp_path / "o" / "i", "triton": tmp_path / "o" / "t"}
    with pytest.raises(tarfile.OutsideDestinationError):
        cb.unpack(bundle, KEY, out, log=lambda _: None)
    assert not (tmp_path / "o" / "escaped").exists()


def test_cli_wires_cache_subcommands(monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr(cb, "stack_key", lambda device=None: KEY)
    monkeypatch.setattr(cb, "cache_dirs", lambda: {"inductor": tmp_path / "i", "triton": tmp_path / "t"})

    def pack(root, key, dirs, kinds=None, note=None):
        calls["pack"] = (root, dirs, kinds, note)
        return root, []

    monkeypatch.setattr(cb, "pack", pack)
    args = build_parser().parse_args(["cache", "pack", str(tmp_path / "r"), "--kinds", "triton", "--note", "n"])
    args.func(args)
    assert calls["pack"] == (tmp_path / "r", {"inductor": tmp_path / "i", "triton": tmp_path / "t"}, ["triton"], "n")

    def refuse(*a, **k):
        raise cb.BundleMismatchError("nothing for this stack")

    monkeypatch.setattr(cb, "unpack", refuse)
    args = build_parser().parse_args(["cache", "unpack", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == cache_cli.EXIT_NO_MATCH
    args = build_parser().parse_args(["serve", "qwen3_5_9b", "--compile-bundle", str(tmp_path)])
    assert args.compile_bundle == str(tmp_path)
