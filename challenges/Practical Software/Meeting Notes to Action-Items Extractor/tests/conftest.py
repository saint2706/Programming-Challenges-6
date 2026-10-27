"""Shared fixtures. Tests never need the network or a model; the LLM server is a local fake."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def llm_server():
    """A loopback chat endpoint whose status and body the test controls.

    Yields (state, url). ``state["requests"]`` collects the JSON payloads received.
    """
    state: dict = {"status": 200, "body": b"", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            state["requests"].append(json.loads(self.rfile.read(length) or b"{}"))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
