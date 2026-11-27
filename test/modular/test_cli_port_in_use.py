"""``mstar serve`` refuses a port that is already taken before it loads anything."""

import socket
import sys
import threading
import types

import pytest

from mstar.cli import main as cli


@pytest.fixture
def launched(monkeypatch):
    calls: list[list[str]] = []
    fake = types.ModuleType("mstar.api_server.entrypoint")
    fake.main = calls.append
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake)
    monkeypatch.setattr(cli, "_healthy", lambda url, timeout_s=5.0: False, raising=False)
    monkeypatch.delenv("MSTAR_SERVER_HOST", raising=False)
    return calls


@pytest.fixture
def listener():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        yield s.getsockname()[1]


def test_a_taken_port_is_refused_before_the_server_starts(launched, listener):
    with pytest.raises(SystemExit) as exited:
        cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--host", "127.0.0.1", "--port", str(listener)]))
    assert f"error: port {listener} is already in use" in str(exited.value)
    assert f"ss -ltnp | grep :{listener}" in str(exited.value) and "another --port" in str(exited.value)
    assert launched == [], "the server started on a port it could never bind"


def test_the_checks_use_the_address_the_frontend_binds(launched, monkeypatch):
    # mstar-server binds an exported MSTAR_SERVER_HOST over --host; the Python frontend binds --host
    checked = []
    monkeypatch.setattr(cli, "_port_in_use", lambda host, port: checked.append(host) or False)
    monkeypatch.setattr(cli, "_announce_when_healthy", lambda url, banner: checked.append(url) or threading.Event())
    monkeypatch.setenv("MSTAR_SERVER_HOST", "127.0.0.2")
    for extra in ([], ["--python-frontend"]):
        cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--port", "8123", *extra]))
    assert checked == ["127.0.0.2", "http://127.0.0.2:8123/health", "0.0.0.0", "http://localhost:8123/health"]
    assert [argv[argv.index("--host") + 1] for argv in launched] == ["0.0.0.0", "0.0.0.0"]


def test_a_wildcard_bind_sees_a_port_taken_on_one_address(listener):
    if sys.platform != "linux":
        pytest.skip("BSD lets a wildcard bind share a port held on one address, as the server's own bind would")
    assert cli._port_in_use("0.0.0.0", listener)


def test_a_free_port_is_not_in_use():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert not cli._port_in_use("127.0.0.1", port)


@pytest.mark.parametrize(("host", "port"), [("no-such-host.invalid", 8000), ("127.0.0.1", 70000)])
def test_a_host_or_port_the_server_cannot_bind_is_left_to_its_own_error(host, port):
    assert not cli._port_in_use(host, port)
