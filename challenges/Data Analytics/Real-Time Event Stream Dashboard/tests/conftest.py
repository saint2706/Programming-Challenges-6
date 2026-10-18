"""Fixtures.

Which Redis the tests talk to, in order:

1. ``STREAM_TEST_REDIS_URL`` (the author's runs use a local Memurai 7.2 on Windows).
2. A ``redis-server`` / ``valkey-server`` found on PATH, started on a free port for the session (CI
   installs one when ``pyproject.toml`` depends on redis).
3. Neither: the ``broker`` fixture falls back to an in-process fakeredis, so the pipeline tests still
   run, and the tests that need a URL (separate processes, the CLI, the Streamlit app) are skipped.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import redis
from stream_dashboard.broker import Broker, connect
from stream_dashboard.events import load_events
from stream_dashboard.paths import ENV_VAR
from stream_dashboard.windows import Layout

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "events.csv.gz"


@pytest.fixture(scope="session", autouse=True)
def _home():
    os.environ[ENV_VAR] = str(ROOT)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_url():
    url = os.environ.get("STREAM_TEST_REDIS_URL")
    if url:
        yield url
        return
    exe = shutil.which("redis-server") or shutil.which("valkey-server")
    if not exe:
        yield None
        return
    port = _free_port()
    proc = subprocess.Popen(
        [
            exe,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    try:
        client = redis.Redis.from_url(url)
        for _ in range(100):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"{exe} did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def redis_url(server_url):
    if server_url is None:
        pytest.skip(
            "needs a Redis server: set STREAM_TEST_REDIS_URL or put redis-server on PATH"
        )
    return server_url


@pytest.fixture
def layout():
    return Layout(prefix=f"sdtest:{uuid.uuid4().hex[:8]}:")


@pytest.fixture
def broker(server_url, layout):
    if server_url:
        client = connect(server_url)
    else:
        import fakeredis

        client = fakeredis.FakeRedis(
            server=fakeredis.FakeServer(), decode_responses=True
        )
    b = Broker(client, layout)
    b.ensure_group()
    yield b
    b.reset()


@pytest.fixture(scope="session")
def events():
    """The first 6000 real events (about 5 minutes of traffic), sorted by event time."""
    return load_events(SAMPLE, limit=6000)
