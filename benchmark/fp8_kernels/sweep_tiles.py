#!/usr/bin/env python3
"""Sweep launch tiles for the fp8 block-scale kernels at a model's shapes on this GPU, and print
the table rows to paste into the kernel's module.

    # dense linears (mstar/utils/dense_fp8.py's _TILES): per-rank (N, K, glu), e.g. GLM-5.2's
    # o_proj and shared-expert gate/up at TP8
    python benchmark/fp8_kernels/sweep_tiles.py dense --shape 6144,2048,0 --shape 256,6144,1

    # small-batch MoE (mstar/utils/fused_moe/decode.py's _TABLES): one TP rank's experts
    python benchmark/fp8_kernels/sweep_tiles.py moe-decode --hidden 7168 --inter 256 \\
        --experts 256 --top-k 8

Each candidate runs inside a CUDA graph (how decode runs in production) and must match the
first candidate's output within bf16 rounding before it competes on time. One idle GPU,
minutes per shape.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
from functools import partial

import torch

from mstar.platforms import current_platform

TOKEN_BUCKETS = (1, 2, 4, 16, 32, 64)


def time_graph(fn, iters: int = 200, capture_n: int = 10) -> float:
    """Microseconds per call of ``fn``, replayed from a CUDA graph of ``capture_n`` calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(capture_n):
            fn()
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * capture_n)


def _random_fp8(rows: int, cols: int, block=(128, 128)):
    w = torch.randn(rows, cols, device="cuda") * 0.02
    bo, bi = block
    nb, kb = -(-rows // bo), -(-cols // bi)
    padded = torch.zeros(nb * bo, kb * bi, device="cuda")
    padded[:rows, :cols] = w
    b = padded.view(nb, bo, kb, bi)
    s = b.abs().amax(dim=(1, 3)) / 448.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (b / s[:, None, :, None]).to(torch.float8_e4m3fn).view(nb * bo, kb * bi)[:rows, :cols]
    return q.contiguous().view(torch.uint8), s.contiguous()


def _dense(dense_fp8, x, q, s, block, glu, cfg):
    return dense_fp8._w8a16(x, q, s, block, glu, dict(cfg))


@contextlib.contextmanager
def _tuned(decode, tuning):
    """Every decode launch takes ``tuning`` while the block runs."""
    orig = decode._tuning
    decode._tuning = lambda device, hidden: tuning
    try:
        yield
    finally:
        decode._tuning = orig


def _experts(decode, x, w13, s13, w2, s2, sw13, sw2, tw, ids):
    return decode.experts(x, w13, s13, w2, s2, sw13, sw2, tw, ids, block_size=(128, 128))


def sweep_dense(shapes: list[tuple[int, int, bool]]) -> None:
    from mstar.utils import dense_fp8

    dense_fp8.reserve("cuda")
    arch = current_platform.arch()
    block = (128, 128)
    for n, k, glu in shapes:
        q, s = _random_fp8(2 * n if glu else n, k)
        rows = {}
        for tokens in TOKEN_BUCKETS:
            x = (torch.randn(tokens, k, device="cuda") * 0.5).bfloat16()
            best, ref = None, None
            for dot, bn, bk, split, stages, warps in itertools.product(
                    (False, True), (8, 16, 32, 64), (128, 256, 512), (1, 2, 4, 8, 16),
                    (2, 3, 4, 5), (1, 2, 4)):
                if not dot and tokens > 4:
                    continue
                cfg = dict(DOT=dot, BN=bn, BK=bk, SPLIT=split, STAGES=stages, num_warps=warps)
                run = partial(_dense, dense_fp8, x, q, s, block, glu, cfg)
                try:
                    out = run()
                    if ref is None:
                        ref = out
                    elif not torch.allclose(out.float(), ref.float(), rtol=2**-7, atol=1e-3):
                        continue
                    us = time_graph(run)
                except Exception:  # noqa: BLE001 - a config the kernel's asserts or Triton reject
                    continue
                if best is None or us < best[0]:
                    best = (us, (dot, bn, bk, split, stages, warps))
            rows[tokens] = best
            print(f"N {n} K {k} glu {glu} tokens {tokens}: {best[0]:.2f} us {best[1]}", flush=True)
        entries = ", ".join(
            f"{t}: ({'_DOT' if c[0] else '_FMA'}, {', '.join(map(str, c[1:]))})"
            for t, (_, c) in rows.items())
        print(f'    ("{arch}", {n}, {k}, {glu}): {{{entries}}},')


def sweep_moe_decode(hidden: int, inter: int, experts: int, top_k: int) -> None:
    from mstar.utils.fused_moe import decode

    arch = current_platform.arch()
    w13, s13 = zip(*(_random_fp8(2 * inter, hidden) for _ in range(experts)), strict=True)
    w2, s2 = zip(*(_random_fp8(hidden, inter) for _ in range(experts)), strict=True)
    w13, s13, w2, s2 = (torch.stack(t) for t in (w13, s13, w2, s2))
    sw13 = (torch.randn(2 * inter, hidden, device="cuda") * 0.02).bfloat16()
    sw2 = (torch.randn(hidden, inter, device="cuda") * 0.02).bfloat16()
    base = decode._tuning(torch.device("cuda", torch.cuda.current_device()), hidden)
    gate_up_grid = [dict(bj=bj, bk=bk, bjs=bjs, bks=bks, warps=w) for bj, bk, bjs, bks, w in
                    itertools.product((8, 16), (256, 512, 1024), (16, 32), (128, 256), (2, 4))]
    down_grid = [dict(bn=bn, bc=bc, warps=w) for bn, bc, w in
                 itertools.product((16, 32), (128, 256), (4, 8))]
    pair_best = {}
    for tokens in range(1, 17):
        x = torch.randn(tokens, hidden, device="cuda").bfloat16()
        ids = torch.stack([torch.randperm(experts, device="cuda")[:top_k] for _ in range(tokens)])
        tw = torch.rand(tokens, top_k, device="cuda")
        run = partial(_experts, decode, x, w13, s13, w2, s2, sw13, sw2, tw, ids)
        timings = []
        for gu, dn in itertools.product(gate_up_grid, down_grid):
            tuning = decode._Tuning(pair_max_tokens=16, gate_up=((None, gu),),
                                    down=((None, dn),), router=base.router)
            try:
                with _tuned(decode, tuning):
                    us = time_graph(run)
            except Exception:  # noqa: BLE001 - a tile Triton rejects
                continue
            timings.append((us, gu, dn))
        grouped_tuning = decode._Tuning(pair_max_tokens=0, gate_up=base.gate_up, down=base.down,
                                        router=base.router)
        with _tuned(decode, grouped_tuning):
            grouped = time_graph(run)
        best = min(timings, key=lambda t: t[0])
        pair_best[tokens] = (best, grouped)
        print(f"tokens {tokens}: per-pair {best[0]:.2f} us {best[1]} {best[2]}, "
              f"grouped {grouped:.2f} us", flush=True)
    pair_max = max((t for t, (best, grouped) in pair_best.items() if best[0] < grouped), default=0)
    print(f"# {arch}, hidden {hidden}: pair_max_tokens={pair_max}; take the rows above as the "
          f"gate_up / down (max tokens, tiles) table of _TABLES[('{arch}', {hidden})]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    dense = sub.add_parser("dense")
    dense.add_argument("--shape", action="append", required=True,
                       help="N,K,glu of one per-rank linear (glu 0 or 1)")
    moe = sub.add_parser("moe-decode")
    moe.add_argument("--hidden", type=int, required=True)
    moe.add_argument("--inter", type=int, required=True, help="per-rank moe intermediate")
    moe.add_argument("--experts", type=int, required=True)
    moe.add_argument("--top-k", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "dense":
        shapes = []
        for spec in args.shape:
            n, k, glu = (int(v) for v in spec.split(","))
            shapes.append((n, k, bool(glu)))
        sweep_dense(shapes)
    else:
        sweep_moe_decode(args.hidden, args.inter, args.experts, args.top_k)


if __name__ == "__main__":
    main()
