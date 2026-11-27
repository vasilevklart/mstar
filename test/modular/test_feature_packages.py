"""A model whose runtime package no extra installs refuses to start without it, before
any worker starts, and names the package and the line that installs it."""

import re
import sys
from pathlib import Path

import pytest
import yaml

from mstar.model.omnivoice import omnivoice_model
from mstar.model.omnivoice.omnivoice_model import OMNIVOICE_INSTALL, OmniVoiceModel
from mstar.validation import DeploymentError, validate_deployment

ROOT = Path(__file__).resolve().parents[2]


def _validate(model, config_name):
    path = ROOT / "configs" / config_name
    return validate_deployment(model, yaml.safe_load(path.read_text()), str(path))


@pytest.fixture
def no_omnivoice(monkeypatch):
    monkeypatch.setitem(sys.modules, "omnivoice", None)


@pytest.fixture
def pypi_omnivoice(tmp_path, monkeypatch):
    """omnivoice as PyPI's release ships it: the package and its models, no omnivoice_flashinfer."""
    models = tmp_path / "omnivoice" / "models"
    models.mkdir(parents=True)
    (tmp_path / "omnivoice" / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    ours = [name for name in sys.modules if name.split(".")[0] == "omnivoice"]
    before = {name: sys.modules.pop(name) for name in ours}
    yield
    for name in [name for name in sys.modules if name.split(".")[0] == "omnivoice"]:
        del sys.modules[name]
    sys.modules.update(before)


def test_omnivoice_without_its_package_is_refused_before_any_worker_starts(no_omnivoice):
    with pytest.raises(DeploymentError, match="no worker was started") as refused:
        _validate(OmniVoiceModel("k2-fsa/OmniVoice"), "omnivoice.yaml")
    assert [p for p in refused.value.problems if "omnivoice package" in p and OMNIVOICE_INSTALL in p]


def test_omnivoice_from_pypi_is_refused_for_the_packed_forward_it_lacks(pypi_omnivoice):
    problem = omnivoice_model.omnivoice_problem()
    assert "omnivoice_flashinfer" in problem and OMNIVOICE_INSTALL in problem
    with pytest.raises(DeploymentError, match="omnivoice_flashinfer"):
        _validate(OmniVoiceModel("k2-fsa/OmniVoice"), "omnivoice.yaml")


def test_omnivoice_in_dummy_mode_needs_no_package(no_omnivoice):
    path = ROOT / "configs" / "omnivoice.yaml"
    OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True).validate_config_yaml(
        yaml.safe_load(path.read_text()), str(path),
    )


def test_omnivoice_workers_name_the_install_too(no_omnivoice):
    """The data worker's warmup and a worker's load say the same, not a bare
    ``No module named 'omnivoice'``."""
    model = OmniVoiceModel("k2-fsa/OmniVoice")
    with pytest.raises(ImportError, match=re.escape(OMNIVOICE_INSTALL)):
        model.warmup_preprocess()
    with pytest.raises(ImportError, match=re.escape(OMNIVOICE_INSTALL)):
        model.get_submodule("backbone")
