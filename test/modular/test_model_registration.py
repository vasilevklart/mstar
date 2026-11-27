"""Every registered model is reachable: CLI default config, and a row in the model matrix.

A model can be entirely correct and still unusable because one of the
registration sites was missed. The matrix in docs/mintlify/models.mdx is what
users read to pick a key, a config and the GPUs for it, so its rows are checked
against the CLI, the configs and pyproject.toml. These checks are pure
filesystem and string work -- no model module is imported -- so they run on CPU
with no weights.
"""
import re
import tomllib
from pathlib import Path

import yaml

from mstar.cli.main import DEFAULT_CONFIGS
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/mintlify/models.mdx"


def _default_configs() -> dict[str, str]:
    return DEFAULT_CONFIGS


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _matrix() -> dict[str, dict[str, str]]:
    """Each row of a pipe table whose header starts with ``Key``, as {header: cell},
    by the registry key its first cell holds in backticks."""
    rows: dict[str, dict[str, str]] = {}
    header = None
    for number, line in enumerate(MATRIX.read_text().splitlines(), 1):
        if not line.startswith("|"):
            header = None
            continue
        cells = _cells(line)
        if header is None:
            header = cells
            continue
        if header[0] != "Key" or set("".join(cells)) <= set("-: "):
            continue  # another table, or the separator row
        where = f"docs/mintlify/models.mdx:{number}"
        assert len(cells) == len(header), f"{where}: {len(cells)} cells under {len(header)} columns"
        key = re.fullmatch(r"`([^`]+)`", cells[0])
        assert key, f"{where}: the first cell holds one registry key in backticks, not {cells[0]!r}"
        assert key[1] not in rows, f"{where}: a second row for {key[1]}"
        rows[key[1]] = dict(zip(header, cells, strict=True))
    return rows


def _gpus(config: str) -> int:
    """The devices a config places workers on: its node groups' ranks after ``rank_devices``."""
    cfg = yaml.safe_load((ROOT / "configs" / config).read_text())
    ranks = {rank for group in cfg["node_groups"] for rank in group["ranks"]}
    devices = cfg.get("rank_devices") or {}
    return len({devices.get(rank, rank) for rank in ranks})


def test_registry_entries_resolve_on_disk():
    """Each (module, class) tuple names a file that defines that class."""
    for name, (module, cls) in MODEL_REGISTRY.items():
        path = ROOT / (module.replace(".", "/") + ".py")
        assert path.is_file(), f"{name}: no module at {module}"
        assert re.search(rf"^class {cls}\b", path.read_text(), re.M), (
            f"{name}: {module} does not define {cls}"
        )


def test_every_model_has_a_cli_default_config():
    """Without a DEFAULT_CONFIGS entry, `mstar serve <name>` exits 'unknown model'."""
    configs = _default_configs()
    missing = sorted(set(MODEL_REGISTRY) - set(configs))
    assert not missing, f"add to DEFAULT_CONFIGS in mstar/cli/main.py: {missing}"
    # The CLI namespace may hold deployment aliases beyond the registry, but
    # every config it names has to exist.
    for name, config in configs.items():
        assert (ROOT / "configs" / config).is_file(), f"{name}: missing configs/{config}"


def test_every_model_is_in_the_matrix():
    """A row per `mstar serve` key and registry key, and none for a key the CLI doesn't take."""
    keys = set(_default_configs()) | set(MODEL_REGISTRY)
    rows = set(_matrix())
    assert not keys - rows, f"add a row to docs/mintlify/models.mdx: {sorted(keys - rows)}"
    assert not rows - keys, f"docs/mintlify/models.mdx has rows for unknown keys: {sorted(rows - keys)}"


def test_matrix_default_config_is_the_cli_default():
    configs = _default_configs()
    wrong = {
        key: (row["Default config"], configs[key])
        for key, row in _matrix().items()
        if key in configs and row["Default config"].strip("`") != configs[key]
    }
    assert not wrong, f"Default config cells in docs/mintlify/models.mdx (cell, DEFAULT_CONFIGS): {wrong}"


def test_matrix_gpus_match_the_default_config():
    configs = _default_configs()
    wrong = {
        key: (row["GPUs"], _gpus(configs[key]))
        for key, row in _matrix().items()
        if key in configs and row["GPUs"] != str(_gpus(configs[key]))
    }
    assert not wrong, f"GPUs cells in docs/mintlify/models.mdx (cell, the config's devices): {wrong}"


def test_matrix_extras_exist():
    """An Extra cell names extras pyproject.toml defines, in backticks, or says none."""
    extras = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"])
    wrong = {}
    for key, row in _matrix().items():
        named = re.findall(r"`([^`]+)`", row["Extra"])
        if (not named and row["Extra"] != "none") or set(named) - extras:
            wrong[key] = row["Extra"]
    assert not wrong, f"Extra cells in docs/mintlify/models.mdx that name no extra in pyproject.toml: {wrong}"


def test_hf_models_are_registered():
    unknown = sorted(set(HF_MODELS) - set(MODEL_REGISTRY))
    assert not unknown, f"HF_MODELS names unregistered models: {unknown}"
