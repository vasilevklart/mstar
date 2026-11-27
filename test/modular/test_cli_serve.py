"""``mstar serve`` argument wiring, with the server launch stubbed out."""

import sys
import types

import pytest

from mstar.cli import main as cli


@pytest.fixture
def launched(monkeypatch):
    calls: list[list[str]] = []
    fake = types.ModuleType("mstar.api_server.entrypoint")
    fake.main = calls.append
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake)
    return calls


def _socket_prefix(argv: list[str]) -> str:
    return argv[argv.index("--socket-path-prefix") + 1]


def test_servers_on_different_ports_get_different_socket_prefixes(launched):
    for port in ("8001", "8002"):
        cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--port", port]))
    first, second = (_socket_prefix(argv) for argv in launched)
    assert first != second and first.endswith("_8001/") and second.endswith("_8002/")


def test_an_explicit_socket_prefix_is_kept(launched):
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--socket-path-prefix", "/tmp/mine/"]))
    assert _socket_prefix(launched[0]) == "/tmp/mine/"


@pytest.mark.parametrize("flags", [
    ["--enable-metrics"], ["--enable-nvtx"], ["--timeout", "1800.0"], ["--tcp-transfer-device", "eth1"],
])
def test_serve_passes_its_flags_to_the_server(launched, flags):
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro", *flags]))
    argv = launched[0]
    assert argv[argv.index(flags[0]):argv.index(flags[0]) + len(flags)] == flags


def test_serve_leaves_the_server_defaults_alone(launched):
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro"]))
    assert not {"--enable-nvtx", "--timeout", "--tcp-transfer-device"} & set(launched[0])


def test_an_explicit_host_beats_an_inherited_one(launched, monkeypatch):
    monkeypatch.setenv("MSTAR_SERVER_HOST", "10.0.0.5")
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--host", "127.0.0.1"]))
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro"]))
    monkeypatch.delenv("MSTAR_SERVER_HOST")
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro"]))
    assert [argv[argv.index("--host") + 1] for argv in launched] == ["127.0.0.1", "10.0.0.5", "0.0.0.0"]


def test_the_launcher_hands_mstar_server_the_resolved_host():
    """mstar-server binds MSTAR_SERVER_HOST; a ``setdefault`` there let an inherited value beat ``--host``."""
    import inspect

    from mstar.api_server import entrypoint

    source = inspect.getsource(entrypoint.main)
    assert "args.host = resolve_host(args.host)" in source
    assert "os.environ[SERVER_HOST_ENV] = args.host" in source


def test_mstar_serve_hides_the_mooncake_port_it_never_used(capsys):
    from mstar.api_server import entrypoint

    with pytest.raises(SystemExit):
        entrypoint.main(["--help"])
    assert "--mooncake-port" not in capsys.readouterr().out, "the help offers a port nothing reads"
    # still accepted, so existing commands keep running
    with pytest.raises(SystemExit):
        entrypoint.main(["--config", "c.yaml", "--mooncake-port", "9000", "--bogus"])
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err
