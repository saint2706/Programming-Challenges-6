"""An in-process SMTP server for tests, so no test ever sends real mail.

Behaviour is scripted: ``replies`` is consumed one entry per DATA command (``None`` = accept),
which lets a test say "fail twice with 451, then succeed".
"""

import asyncio
import datetime as dt
import socket
import ssl
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from pathlib import Path

from aiosmtpd.controller import Controller
from aiosmtpd.smtp import AuthResult


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Handler:
    def __init__(self, replies=(), refuse_rcpt=(), delay=0.0):
        self.delay = delay
        self.replies = list(replies)
        self.refuse_rcpt = set(refuse_rcpt)
        self.messages = []  # parsed email.message.Message objects
        self.envelopes = []
        self.data_calls = 0

    async def handle_RCPT(self, server, session, envelope, address, rcpt_options):
        if address in self.refuse_rcpt:
            return "550 5.1.1 no such user"
        envelope.rcpt_tos.append(address)
        return "250 OK"

    async def handle_DATA(self, server, session, envelope):
        self.data_calls += 1
        if self.delay:
            await asyncio.sleep(
                self.delay
            )  # keeps the send in flight for concurrency tests
        reply = self.replies.pop(0) if self.replies else None
        if reply:
            return reply
        self.envelopes.append(envelope)
        self.messages.append(
            message_from_bytes(envelope.content, policy=policy.default)
        )
        return "250 OK"


@dataclass
class Server:
    handler: Handler
    port: int
    host: str = "127.0.0.1"
    extra: dict = field(default_factory=dict)

    @property
    def messages(self):
        return self.handler.messages


@contextmanager
def smtp_server(*, replies=(), refuse_rcpt=(), credentials=None, tls=None, delay=0.0):
    """``credentials=(user, password)`` demands login; ``tls=(certfile, keyfile)`` offers (and
    requires) STARTTLS."""
    handler = Handler(replies, refuse_rcpt, delay)
    port = free_port()
    kwargs = {}
    if credentials:
        user, password = credentials

        def authenticator(server, session, envelope, mechanism, auth_data):
            ok = (
                getattr(auth_data, "login", b"") == user.encode()
                and getattr(auth_data, "password", b"") == password.encode()
            )
            return AuthResult(success=ok, handled=False)

        kwargs.update(
            authenticator=authenticator, auth_required=True, auth_require_tls=False
        )
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*tls)
        kwargs.update(tls_context=context, require_starttls=True)
    with warnings.catch_warnings():
        # AUTH without TLS is deliberate here: the server is on loopback and exists only for tests.
        warnings.filterwarnings("ignore", message="Requiring AUTH")
        controller = Controller(handler, hostname="127.0.0.1", port=port, **kwargs)
    controller.start()
    try:
        yield Server(handler, port)
    finally:
        controller.stop()


def make_test_ca(directory: Path) -> tuple[Path, Path, Path]:
    """A throwaway CA and a server certificate for 127.0.0.1/localhost signed by it.

    Returns ``(ca_pem, server_cert_pem, server_key_pem)``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    def name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(name("Test CA"))
        .issuer_name(name("Test CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # Python verifies with VERIFY_X509_STRICT, which insists on these extensions.
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    import ipaddress

    cert = (
        x509.CertificateBuilder()
        .subject_name(name("localhost"))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([x509.OID_SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    ca_path, cert_path, key_path = (
        directory / n for n in ("ca.pem", "server.pem", "server.key")
    )
    ca_path.write_bytes(ca_cert.public_bytes(pem))
    cert_path.write_bytes(cert.public_bytes(pem))
    key_path.write_bytes(
        key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_path, cert_path, key_path
