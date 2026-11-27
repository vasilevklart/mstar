"""``mstar serve`` says "starting" at launch and "serving" only once /health answers healthy."""

import json
import sys
import threading
import time
import types
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mstar.cli import main as cli

SERVING = "✓ mstar serving 'kokoro'"


@contextmanager
def health_server():
    """A local /health that answers 503 until ``server.healthy`` is set."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            healthy = self.server.healthy
            body = json.dumps({"status": "healthy" if healthy else "unhealthy"}).encode()
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.healthy = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(cli, "HEALTH_POLL_S", 0.02, raising=False)


def _serve_with(monkeypatch, port, fake_main):
    fake = types.ModuleType("mstar.api_server.entrypoint")
    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake)
    # the stand-in /health holds the port the launch would otherwise refuse
    monkeypatch.setattr(cli, "_port_in_use", lambda host, port: False, raising=False)
    monkeypatch.delenv("MSTAR_SERVER_HOST", raising=False)  # mstar-server would bind that instead
    cli._serve(cli.build_parser().parse_args(["serve", "kokoro", "--host", "127.0.0.1", "--port", str(port)]))


def test_serving_prints_only_once_health_answers(monkeypatch, capsys, fast_polls):
    with health_server() as server:
        port = server.server_address[1]
        seen = {"err": ""}

        def fake_main(argv):
            seen["launch"] = capsys.readouterr().err
            time.sleep(0.2)  # several polls against a 503
            seen["before_healthy"] = capsys.readouterr().err
            server.healthy = True

            def banner_printed():
                seen["err"] += capsys.readouterr().err
                return SERVING in seen["err"]

            seen["printed"] = _wait_for(banner_printed)

        _serve_with(monkeypatch, port, fake_main)

    assert f"starting 'kokoro'; it listens on http://127.0.0.1:{port} once it is ready" in seen["launch"]
    assert "serving" not in seen["launch"] + seen["before_healthy"]
    assert seen["printed"], "no serving banner after /health turned healthy"
    assert f"{SERVING} on http://127.0.0.1:{port}" in seen["err"]
    assert "client.tts(" in seen["err"]


def test_a_server_that_exits_before_it_is_healthy_never_says_serving(monkeypatch, capsys, fast_polls):
    with health_server() as server:
        _serve_with(monkeypatch, server.server_address[1], lambda argv: None)  # setup failed
        server.healthy = True
        time.sleep(0.2)
    assert "starting 'kokoro'" in (err := capsys.readouterr().err)
    assert "serving" not in err


def test_health_check_reads_the_status_and_ignores_proxies(monkeypatch):
    with health_server() as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        assert not cli._healthy(url)  # 503
        server.healthy = True
        # a proxy that would refuse every request must not be asked
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        assert cli._healthy(url)
        port = server.server_address[1]
    assert not cli._healthy(f"http://127.0.0.1:{port}/health")  # nothing listens


@pytest.mark.parametrize(("host", "client"), [
    ("0.0.0.0", "localhost"), ("::", "localhost"), ("127.0.0.1", "127.0.0.1"), ("::1", "[::1]"),
])
def test_the_url_names_a_host_a_client_can_reach(host, client):
    assert cli._base_url(host, 8000) == f"http://{client}:8000"
