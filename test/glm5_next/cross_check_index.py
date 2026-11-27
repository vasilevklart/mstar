"""Cross-check the glm5_next weight mapping against a GLM-5.3-Flash checkpoint's
safetensors index (names only, no tensor data).

    PYTHONPATH=. python test/glm5_next/cross_check_index.py <checkpoint_dir>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.weight_loader import (
    _LAYER_RE,
    expected_parameter_paths,
    resolve_index_names,
)


def cross_check_index(
    index_path: str | Path,
    config: Glm5NextModelConfig | None = None,
    load_mtp: bool = True,
    max_examples: int = 10,
) -> bool:
    """Dry-run the load pipeline over the safetensors index and diff it
    against the expected module tree. Prints a report; returns pass/fail.
    """
    index_path = Path(index_path)
    if index_path.is_dir():
        index_path = index_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    index_names = list(index["weight_map"])

    if config is None:
        with open(index_path.parent / "config.json") as f:
            config = Glm5NextModelConfig.from_hf_config(json.load(f))

    resolved = resolve_index_names(index_names, config, load_mtp=load_mtp)
    loaded_counts: Counter[str] = Counter(
        entry.rsplit(" -> ", 1)[1] for entry in resolved["loaded"]
    )
    absorbed_targets = {
        entry.rsplit(" -> ", 1)[1] for entry in resolved["absorbed"]
    }
    expected = expected_parameter_paths(config, load_mtp=load_mtp)

    unexpected = sorted(set(loaded_counts) - set(expected))
    missing = sorted(set(expected) - set(loaded_counts))
    stray_absorbed = sorted(absorbed_targets - set(expected))
    count_mismatches = sorted(
        f"{target}: got {loaded_counts[target]} shards, want {want}"
        for target, want in expected.items()
        if target in loaded_counts and loaded_counts[target] != want
    )
    non_vision_skips = list(resolved["skip_mtp"])

    n_vision = len(resolved["skip_vision"])
    n_mtp_keys = sum(
        1 for n in index_names
        if (m := _LAYER_RE.match(n)) and int(m.group(1)) >= config.num_hidden_layers
    )
    n_trunk = len(index_names) - n_vision - n_mtp_keys - 1  # - lm_head

    def show(label: str, items: list[str]) -> None:
        print(f"  {label}: {len(items)}")
        for item in items[:max_examples]:
            print(f"    {item}")
        if len(items) > max_examples:
            print(f"    ... and {len(items) - max_examples} more")

    print(f"glm5_next weight-map cross-check: {index_path}")
    print(
        f"  config: {config.num_hidden_layers} layers "
        f"({len(config.kda_layer_indices)} KDA / "
        f"{len(config.full_attn_layer_indices)} MLA+DSA), "
        f"{config.n_routed_experts}+{config.n_shared_experts} experts, "
        f"fp8={config.quantization_config is not None} "
        f"(experts resident={config.moe_fp8_resident}), "
        f"load_mtp={load_mtp}"
    )
    print(
        f"  index: {len(index_names)} tensors = {n_trunk} text trunk "
        f"+ {n_mtp_keys} MTP layer-{config.num_hidden_layers} "
        f"+ {n_vision} vision + 1 lm_head"
    )
    print(
        f"  mapped: {len(resolved['loaded'])} tensors -> "
        f"{len(loaded_counts)} parameters "
        f"(+ {len(resolved['absorbed'])} fp8 scales dequant-absorbed into "
        f"their .weight)"
    )
    skip_note = (
        "OK (vision-only)" if not non_vision_skips
        else f"NOT vision-only ({len(non_vision_skips)} flag-skipped)"
    )
    print(
        f"  skipped: {n_vision + len(non_vision_skips)} "
        f"(vision {n_vision}, mtp {len(resolved['skip_mtp'])}) — "
        f"skip list {skip_note}"
    )
    show("unmapped checkpoint keys", list(resolved["unmapped"]))
    show("unexpected targets (mapped but not in the module tree)", unexpected)
    show("missing targets (module tree got no tensor)", missing)
    show("absorbed scales with no expected .weight target", stray_absorbed)
    show("shard-count mismatches", count_mismatches)

    ok = not (
        resolved["unmapped"] or unexpected or missing
        or stray_absorbed or count_mismatches
    )
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-check the glm5_next weight mapping against a "
        "GLM-5.3-Flash model.safetensors.index.json",
    )
    parser.add_argument(
        "checkpoint",
        help="checkpoint directory (or the index file itself); config.json "
        "must sit next to the index",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="account layer 45 as skipped (mtp_num_draft_tokens == 0 load)",
    )
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    ok = cross_check_index(
        args.checkpoint,
        load_mtp=not args.no_mtp,
        max_examples=args.examples,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
