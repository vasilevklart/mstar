"""What a missing ``mstar_rust`` extension's error tells the user to run."""

import sys

import pytest

from mstar.graph.runtime.utils import require_rust_extension


@pytest.fixture
def no_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, "mstar_rust", None)


def _message(rust_dir) -> str:
    with pytest.raises(ImportError) as raised:
        require_rust_extension(rust_dir)
    return str(raised.value)


def test_a_checkout_gets_the_build_command_for_its_own_crate(no_extension, tmp_path):
    crate = tmp_path / "rust"
    crate.mkdir()
    (crate / "Cargo.toml").write_text('[lib]\nname = "mstar_rust"\n')
    message = _message(crate)
    assert f"`uv pip install maturin && maturin develop --release -m {crate / 'Cargo.toml'}`" in message
    assert "https://rustup.rs" in message


def test_an_installed_package_is_told_to_build_from_a_source_checkout(no_extension, tmp_path):
    message = _message(tmp_path / "rust")  # a wheel has no rust/ beside mstar/
    assert "source checkout (https://github.com/mstar-project/mstar)" in message
    assert "`uv pip install maturin && maturin develop --release -m rust/Cargo.toml`" in message


def test_the_message_does_not_send_users_to_pypi_for_the_extension(no_extension, tmp_path):
    # PyPI's mstar-ai wheel is pure Python (py3-none-any) and has no mstar_rust in it
    message = _message(tmp_path / "rust")
    assert "PyPI" not in message
    assert "pure-Python mstar-ai wheel (py3-none-any) does not include it" in message
