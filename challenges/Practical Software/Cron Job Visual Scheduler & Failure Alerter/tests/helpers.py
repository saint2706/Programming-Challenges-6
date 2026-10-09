"""Shared builders and tiny real servers (HTTP and SMTP) for delivery tests."""

import socketserver
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def at(minutes: float = 0, base: datetime = T0) -> datetime:
    return base + timedelta(minutes=minutes)


class WebhookServer:
    """Records every POST; ``statuses`` is consumed one per request (default 200)."""

    def __init__(self, statuses=()):
        self.requests: list[dict] = []
        self.statuses = list(statuses)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                owner.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                status = owner.statuses.pop(0) if owner.statuses else 200
                self.send_response(status)
                self.end_headers()

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/hook?token=SECRETTOKEN"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


class SMTPServer:
    """A just-enough SMTP server (no TLS, no auth) that keeps each received message."""

    def __init__(self):
        self.messages: list[dict] = []
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def reply(self, line):
                self.wfile.write((line + "\r\n").encode())

            def handle(self):
                self.reply("220 test ESMTP")
                message = {"from": None, "to": [], "data": ""}
                while True:
                    line = self.rfile.readline().decode().rstrip("\r\n")
                    if not line:
                        return
                    command = line.split()[0].upper()
                    if command in {"EHLO", "HELO"}:
                        self.reply("250 test")
                    elif command == "MAIL":
                        message["from"] = line.split(":", 1)[1].strip()
                        self.reply("250 ok")
                    elif command == "RCPT":
                        message["to"].append(line.split(":", 1)[1].strip())
                        self.reply("250 ok")
                    elif command == "DATA":
                        self.reply("354 go")
                        lines = []
                        while (
                            data := self.rfile.readline().decode().rstrip("\r\n")
                        ) != ".":
                            lines.append(data)
                        message["data"] = "\n".join(lines)
                        owner.messages.append(message)
                        message = {"from": None, "to": [], "data": ""}
                        self.reply("250 queued")
                    elif command == "QUIT":
                        self.reply("221 bye")
                        return
                    else:
                        self.reply("250 ok")

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
