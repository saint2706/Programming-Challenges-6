"""The only module that starts processes. Every call is an argv list (no shell), uses hardening
`-c` flags, ignores the user's and system git config, and streams large output instead of
buffering it.

The audited repository is hostile: its config, attributes, hooks, refs, gitfiles, alternates and
commit metadata are treated as data. Records are NUL-delimited everywhere, because NUL is the one
byte no git name, path or identity can contain.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gitaudit.safety import sanitize

MIN_GIT_VERSION = (2, 36)
MAX_RECORD = 4 * 1024**2  # one record longer than this is refused, not buffered
MAX_POINTER_BYTES = 1024**2
_HEX40 = re.compile(rb"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"git version (\d+)\.(\d+)")

# Disable everything in the audited repo's config that could execute code, contact a remote or
# pick a program. `-c` on the command line outranks the repository's own config.
_HARDENING = (
    "-c", "core.fsmonitor=false",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.attributesFile=" + os.devnull,
    "-c", "core.sshCommand=ssh",
    "-c", "core.editor=true",
    "-c", "sequence.editor=true",
    "-c", "credential.helper=",
    "-c", "protocol.allow=never",
    "-c", "gpg.program=gpg",
    "-c", "gpg.openpgp.program=gpg",
    "-c", "gpg.ssh.program=ssh-keygen",
    "-c", "gpg.x509.program=gpgsm",
    "-c", "gpg.format=openpgp",
    "-c", "log.showSignature=false",
    "-c", "color.ui=never",
    "-c", "advice.detachedHead=false",
    "-c", "core.quotePath=false",
    "-c", "i18n.logOutputEncoding=UTF-8",
    "--no-pager",
)  # fmt: skip


class GitError(RuntimeError):
    """A git invocation failed or the repository is unusable. Message is already sanitised."""


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    kind: str  # blob | commit (gitlink) | tree
    sha: str
    size: int | None
    path: str


def _decode_path(raw: bytes) -> str:
    return raw.decode("utf-8", "backslashreplace")


def _hardened_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",
        GIT_NO_LAZY_FETCH="1",
        GIT_PAGER="cat",
        GIT_ASKPASS="",
        LC_ALL="C",
    )
    return env


def _is_network_path(text: str) -> bool:
    """Any doubled slash, UNC and `//host/share` included. Windows opens SMB for those, so they are
    refused rather than resolved. Refusing a harmless doubled slash costs nothing."""
    return "//" in text.replace("\\", "/")


def _read_pointer(path: Path) -> bytes | None:
    """The file's bytes, or None when absent. Over the size limit is refused, not truncated."""
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_POINTER_BYTES:
        raise GitError(
            f"refusing {sanitize(path.name, 40)}: too large for a git pointer"
        )
    return path.read_bytes()


def _same_dir(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _one_line(data: bytes, what: str) -> str:
    """The text of a one-line pointer file. Anything else is refused: no loose parsing."""
    if data.count(b"\n") != 1 or not data.endswith(b"\n"):
        raise GitError(f"refusing {what}: it must be exactly one line")
    try:
        text = data[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GitError(f"refusing {what}: it is not UTF-8") from exc
    if not text or any(ord(ch) < 0x20 for ch in text):
        raise GitError(f"refusing {what}: empty or contains control characters")
    return text


def _gitfile_target(dot: Path) -> str:
    """git's own rule: the file starts with `gitdir: `. Only that one-line form is accepted."""
    data = _read_pointer(dot) or b""
    if not data.startswith(b"gitdir: "):
        raise GitError("refusing a .git file that does not start with 'gitdir: '")
    return _one_line(data[len(b"gitdir: ") :], "the .git file")


def _gitdir_candidates(path: Path) -> list[tuple[Path, Path]]:
    """(worktree root, storage dir) per place git could find a repository above `path`, nearest first.

    Reads only local pointer bytes, so nothing is opened over the network.
    """
    found: list[tuple[Path, Path]] = []
    for c in (path, *path.parents):
        dot = c / ".git"
        if dot.is_dir():
            if not ((dot / "HEAD").is_file() and (dot / "objects").is_dir()):
                raise GitError("refusing a .git directory that is not a repository")
            found.append((c, dot))
        elif dot.is_file():
            target = _gitfile_target(dot)
            if _is_network_path(target):
                raise GitError("refusing a gitfile that points at a network path")
            found.append((c, c / target))
        elif (c / "HEAD").is_file() and (c / "objects").is_dir():
            found.append((c, c))  # a bare repository is its own storage dir
    return found


def _posix_owner_ok(owner_uid: int, euid: int) -> bool:
    """Only the running user's own files pass. Root-owned files are refused unless we are root."""
    return owner_uid == euid


def _windows_owner_ok(path: Path) -> bool:
    """True when the file's owner SID is the current user's SID, read through advapi32 ctypes."""
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared types: untyped handles would be truncated to 32 bits on 64-bit Windows.
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    owner = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    rc = advapi.GetNamedSecurityInfoW(
        str(path), 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(sd)
    )
    if rc:
        raise OSError(rc, "GetNamedSecurityInfoW failed")
    try:
        token = wintypes.HANDLE()
        if not advapi.OpenProcessToken(
            kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            size = wintypes.DWORD()
            advapi.GetTokenInformation(
                token, 1, None, 0, ctypes.byref(size)
            )  # TokenUser
            buf = ctypes.create_string_buffer(size.value)
            if not advapi.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            user_sid = ctypes.c_void_p.from_buffer(
                buf
            ).value  # TOKEN_USER starts with the SID
            return bool(advapi.EqualSid(owner, ctypes.c_void_p(user_sid)))
        finally:
            kernel.CloseHandle(token)
    finally:
        kernel.LocalFree(sd)


def _owner_ok(path: Path) -> bool:
    if os.name != "nt":
        return _posix_owner_ok(path.stat().st_uid, os.geteuid())
    try:
        return _windows_owner_ok(path)
    except OSError as exc:
        raise GitError(
            f"cannot determine the owner of {sanitize(path.name, 40)}; "
            "pass --trust-owner to skip this check"
        ) from exc


def _check_owner(root: Path, gitdir: Path, trust_owner: bool, safe: set[str]) -> None:
    """Refuse a repository the current user does not own, as git's own safe.directory check would.

    The CLI's --safe-directory and --trust-owner are the only ways past this. Nothing is read from
    the audited repository's config for it.
    """
    if trust_owner or _same_dir(root) in safe or _same_dir(gitdir) in safe:
        return
    for path in {root, gitdir}:  # a bare repository has one path
        if not _owner_ok(path):
            raise GitError(
                f"refusing {sanitize(path.name, 40)}: owned by another user "
                "(run as its owner, or pass --safe-directory PATH or --trust-owner)"
            )


def _pinned_env(
    env: dict[str, str], root: Path, gitdir: Path, scratch: str
) -> dict[str, str]:
    """GIT_DIR, and GIT_WORK_TREE always, so core.worktree is never consulted.

    A bare repository gets an empty scratch directory as its work tree.
    """
    work_tree = root if root != gitdir else Path(scratch)
    return dict(env, GIT_DIR=str(gitdir), GIT_WORK_TREE=str(work_tree))


def _refuse_includes(config: Path, env: dict[str, str]) -> None:
    """Refuse a config file that git reads an include from.

    Asks git for the key names with includes not followed, so git's own parser decides what
    counts as an include. A file git cannot parse raises here too.
    """
    proc = subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(config),
            "--no-includes",
            "--list",
            "--name-only",
            "-z",
        ],
        capture_output=True,
        env=env,
        cwd=tempfile.gettempdir(),  # outside any repository: nothing else is read
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(f"git cannot read {sanitize(config.name, 40)}")
    for raw in proc.stdout.split(b"\0"):
        key = raw.decode("utf-8", "replace").lower()
        if key.startswith(("include.", "includeif.")):
            raise GitError(
                f"refusing {sanitize(config.name, 40)}: it uses an include, which this audit "
                "does not follow"
            )


def _refuse_network_worktree(config: Path, env: dict[str, str]) -> None:
    """Refuse a core.worktree that names a network path. Every value is read, not just the last."""
    proc = subprocess.run(
        ["git", "config", "--file", str(config), "--get-all", "-z", "core.worktree"],
        capture_output=True,
        env=env,
        cwd=tempfile.gettempdir(),
        timeout=60,
        check=False,
    )
    if proc.returncode == 1:  # key absent
        return
    if proc.returncode != 0:
        raise GitError(f"git cannot read {sanitize(config.name, 40)}")
    for raw in proc.stdout.split(b"\0"):
        if raw and _is_network_path(raw.decode("utf-8", "replace")):
            raise GitError(
                f"refusing {sanitize(config.name, 40)}: core.worktree is a network path"
            )


def _refuse_untrusted_storage(gitdir: Path, env: dict[str, str]) -> None:
    """Refuse commondir, config-include and alternates tricks in one storage dir, before git reads them."""
    common_data = _read_pointer(gitdir / "commondir")
    common = gitdir.resolve()
    if common_data is not None:
        text = _one_line(common_data, "commondir")
        if _is_network_path(text):
            raise GitError("refusing a commondir that points at a network path")
        common = (gitdir / text).resolve()
    for config in (gitdir / "config", gitdir / "config.worktree", common / "config"):
        if config.is_file():
            _refuse_includes(config, env)
            _refuse_network_worktree(config, env)
    # Alternates are not followed: any entry at all is refused, so no entry is ever parsed.
    alternates = _read_pointer(common / "objects" / "info" / "alternates") or b""
    if alternates.strip(b"\n"):
        raise GitError(
            "refusing object alternates: run git repack -a -d, then remove objects/info/alternates"
        )


class GitRepo:
    """A repository handle. Methods stream where the output can be large."""

    def __init__(
        self,
        path: Path,
        timeout: float = 1800.0,
        *,
        trust_owner: bool = False,
        safe_directory: Iterable[str] = (),
    ) -> None:
        self.path = Path(path).resolve()
        self.timeout = timeout
        base = _hardened_env()
        self._env = base
        self._check_version()
        if not self.path.is_dir():
            raise GitError(f"not a directory: {sanitize(self.path, 200)}")
        candidates = _gitdir_candidates(self.path)
        if not candidates:
            raise GitError("not a git repository")
        root, gitdir = candidates[0]
        # Python-only checks first: no git process runs before the owner check passes.
        _check_owner(root, gitdir, trust_owner, {_same_dir(s) for s in safe_directory})
        self._scratch = tempfile.TemporaryDirectory(prefix="gitaudit-worktree-")
        for c_root, c_gitdir in candidates:
            pinned = _pinned_env(base, c_root, c_gitdir, self._scratch.name)
            _refuse_untrusted_storage(c_gitdir, pinned)
        # The nearest candidate is the one git uses. Pin it, so later git calls cannot pick another.
        self._env = _pinned_env(base, root, gitdir, self._scratch.name)
        self.git_dir = self._discover(gitdir)

    # ----- process plumbing -------------------------------------------------------------

    def _argv(self, args: Iterable[str]) -> list[str]:
        return ["git", "-C", str(self.path), *_HARDENING, *args]

    def _check_version(self) -> None:
        out = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            env=self._env,
            check=False,
        ).stdout
        match = _VERSION_RE.search(out)
        if match is None:
            raise GitError("git is not installed or not on PATH")
        if (int(match.group(1)), int(match.group(2))) < MIN_GIT_VERSION:
            need = ".".join(map(str, MIN_GIT_VERSION))
            raise GitError(
                f"git {need} or newer is required, found {sanitize(out.strip(), 60)}"
            )

    def _discover(self, expected: Path) -> Path:
        """Post-check: git, pinned to GIT_DIR, must report the directory that was validated."""
        out = (
            self.run(["rev-parse", "--absolute-git-dir"])
            .decode("utf-8", "replace")
            .strip()
        )
        if _same_dir(out) != _same_dir(expected):
            raise GitError("git reports a storage directory the audit did not check")
        return Path(out)

    def run(self, args: list[str], stdin: bytes | None = None) -> bytes:
        """Run a small command and return its full stdout."""
        with tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.run(
                    self._argv(args),
                    input=stdin,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    env=self._env,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise GitError(f"git {args[0]} timed out") from exc
            if proc.returncode != 0:
                raise GitError(f"git {args[0]} failed: {self._stderr_text(err)}")
            return proc.stdout

    @staticmethod
    def _stderr_text(err) -> str:
        err.seek(0)
        return sanitize(err.read(400).decode("utf-8", "replace").strip(), 300)

    @contextmanager
    def _stream(self, args: list[str], feed: Iterable[bytes] | None = None):
        """Start a process whose stdout is read incrementally. Feeds stdin from a thread.

        A timer kills git after `self.timeout` seconds; the consumer then sees end of output and
        the stream raises. A consumer that stops early (or raises) kills git on purpose, and
        that is not an error. Only a clean drain whose exit code is nonzero raises.
        """
        err = tempfile.TemporaryFile()  # noqa: SIM115 closed in the finally below
        proc = subprocess.Popen(
            self._argv(args),
            stdin=subprocess.PIPE if feed is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err,
            env=self._env,
        )
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(self.timeout, expire)
        timer.daemon = True
        timer.start()
        writer: threading.Thread | None = None
        if feed is not None:

            def pump() -> None:
                try:
                    for chunk in feed:
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            writer = threading.Thread(target=pump, daemon=True)
            writer.start()
        abandoned = False
        try:
            yield proc
        except BaseException:
            abandoned = True
            raise
        finally:
            timer.cancel()
            if abandoned and proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            if writer is not None:
                writer.join(timeout=5)
            code = proc.wait()
            try:
                if timed_out.is_set():
                    raise GitError(f"git {args[0]} timed out after {self.timeout:g} s")
                if code != 0 and not abandoned:
                    raise GitError(
                        f"git {args[0]} exited {code}: {self._stderr_text(err)}"
                    )
            finally:
                err.close()

    def records(self, args: list[str], sep: bytes = b"\0") -> Iterator[bytes]:
        """Yield sep-delimited records. Pieces are collected in a list and joined once per record,
        so one long record costs linear time, and a record over MAX_RECORD is refused."""
        with self._stream(args) as proc:
            parts: list[bytes] = []
            size = 0
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    break
                start = 0
                while (idx := chunk.find(sep, start)) >= 0:
                    parts.append(chunk[start:idx])
                    size += idx - start
                    if size > MAX_RECORD:
                        raise GitError(f"git {args[0]} produced a record over 4 MiB")
                    yield b"".join(parts)
                    parts, size, start = [], 0, idx + len(sep)
                tail = chunk[start:]
                size += len(tail)
                if size > MAX_RECORD:
                    raise GitError(f"git {args[0]} produced a record over 4 MiB")
                parts.append(tail)
            if size:
                yield b"".join(parts)

    # ----- queries ----------------------------------------------------------------------

    def resolve_commit(self, ref: str) -> str:
        if ref.startswith("-") or not ref:
            raise GitError("refusing a ref that starts with '-'")
        out = self.run(
            ["rev-parse", "--verify", "-q", "--end-of-options", f"{ref}^{{commit}}"]
        )
        sha = out.decode("ascii", "replace").strip()
        if not _HEX40.fullmatch(sha.encode("ascii", "replace")):
            raise GitError(f"cannot resolve ref {sanitize(ref, 80)!r} to a commit")
        return sha

    def ls_tree(self, commit_sha: str) -> list[TreeEntry]:
        """Full recursive tree of a commit with blob sizes. Paths are NUL-delimited, so any name is safe."""
        entries: list[TreeEntry] = []
        for rec in self.records(
            ["ls-tree", "-r", "-l", "-z", "--full-tree", commit_sha]
        ):
            meta, sep, name = rec.partition(b"\t")
            if not sep:
                continue
            mode, kind, sha, size = meta.decode("ascii", "replace").split()
            entries.append(
                TreeEntry(
                    mode=mode,
                    kind=kind,
                    sha=sha,
                    size=None if size == "-" else int(size),
                    path=_decode_path(name),
                )
            )
        return entries

    def for_each_ref(self, pattern: str = "refs/") -> list[dict[str, str]]:
        """Refs with the fields the audit needs. Every field ends in NUL, which no ref name, author
        name or email can contain, so control characters in identities cannot shift columns."""
        fmt = (
            "%(refname)%00%(objectname)%00%(objecttype)%00%(committerdate:unix)%00"
            "%(authorname)%00%(authoremail)%00%(upstream)%00%(upstream:track)%00%(symref)%00"
        )
        out = self.run(["for-each-ref", f"--format={fmt}", pattern])
        keys = (
            "name", "sha", "type", "date", "author", "email", "upstream", "track", "symref",
        )  # fmt: skip
        tokens = out.split(b"\0")
        rows = []
        for i in range(0, len(tokens) - len(keys) + 1, len(keys)):
            group = tokens[i : i + len(keys)]
            group[0] = group[0].lstrip(
                b"\n"
            )  # for-each-ref ends each record with a newline
            rows.append({k: _decode_path(v) for k, v in zip(keys, group, strict=True)})
        return rows

    def merged_into(self, commit_sha: str) -> set[str]:
        out = self.run(
            [
                "for-each-ref",
                f"--merged={commit_sha}",
                "--format=%(objectname)",
                "refs/",
            ]
        )
        return {
            line.strip()
            for line in out.decode("ascii", "replace").splitlines()
            if line.strip()
        }

    def ahead_behind(self, base_sha: str, tip_sha: str) -> tuple[int, int]:
        out = self.run(
            ["rev-list", "--left-right", "--count", f"{base_sha}...{tip_sha}"]
        )
        left, right = out.decode("ascii", "replace").split()
        return int(right), int(left)  # (ahead, behind) relative to base

    def merge_base(self, a: str, b: str) -> str | None:
        proc = subprocess.run(
            self._argv(["merge-base", a, b]),
            capture_output=True, env=self._env, check=False, timeout=self.timeout,
        )  # fmt: skip
        if proc.returncode != 0:
            return None
        sha = proc.stdout.decode("ascii", "replace").strip()
        return sha if _HEX40.fullmatch(sha.encode("ascii", "replace")) else None

    def commit_time(self, sha: str) -> int:
        return int(
            self.run(["show", "-s", "--format=%ct", sha]).decode("ascii").strip()
        )

    def object_blobs(self, since: str | None = None) -> Iterator[str]:
        """Every reachable blob id in every ref (path names not requested)."""
        args = [
            "rev-list",
            "--objects",
            "--no-object-names",
            "--all",
            "--filter=object:type=blob",
        ]
        if since:
            args.append(f"--since={since}")
        for rec in self.records(args, sep=b"\n"):
            if len(rec) == 40:
                yield rec.decode("ascii", "replace")

    def batch_check(self, shas: Iterable[str]) -> Iterator[tuple[str, str, int]]:
        """(sha, type, size) for each object id, streamed in input order. Missing objects are skipped."""
        feed = (f"{sha}\n".encode("ascii") for sha in shas)
        args = ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"]
        with self._stream(args, feed=feed) as proc:
            for line in proc.stdout:
                parts = line.decode("ascii", "replace").split()
                if len(parts) == 3 and parts[2].isdigit():
                    yield parts[0], parts[1], int(parts[2])

    def blob_contents(
        self, shas: Iterable[str], limit: int
    ) -> Iterator[tuple[str, bytes, int]]:
        """(sha, data, size) with data capped at `limit` bytes. Oversized tails are read and
        dropped, so the stream stays in sync without holding a large blob in memory."""
        feed = (f"{sha}\n".encode("ascii") for sha in shas)
        with self._stream(["cat-file", "--batch"], feed=feed) as proc:
            stdout = proc.stdout
            while True:
                header = stdout.readline()
                if not header:
                    return
                parts = header.decode("ascii", "replace").split()
                if len(parts) != 3 or not parts[2].isdigit():
                    continue  # "<sha> missing"
                size = int(parts[2])
                data = stdout.read(min(size, limit))
                remaining = size - len(data)
                while remaining > 0:
                    skipped = stdout.read(min(remaining, 1 << 16))
                    if not skipped:
                        return
                    remaining -= len(skipped)
                stdout.read(1)  # newline that follows every object
                yield parts[0], data, size

    def commits(self, since: str | None = None) -> Iterator[dict[str, object]]:
        """Oldest-first commits (not merges) with the paths each one changed.

        The header is NUL-delimited and read by count (time, author, email, subject), so an
        identity or subject with control characters cannot move a field. Raw entries are parsed
        as a NUL-delimited state machine, so a path with spaces, tabs or newlines is safe too.
        """
        args = [
            "log", "--all", "--no-merges", "--reverse", "--topo-order", "--raw", "-z",
            "--no-renames", "--no-abbrev", "--no-ext-diff", "--no-textconv",
            "--format=%x01%H%x00%ct%x00%an%x00%ae%x00%s",
        ]  # fmt: skip
        if since:
            args.append(f"--since={since}")
        current: dict[str, object] | None = None
        pending: tuple[str, str, str] | None = None
        header_left = 0
        header: list[bytes] = []
        for tok in self.records(args):
            if header_left:
                header.append(tok)
                header_left -= 1
                if header_left == 0:
                    ts, author, email, subject = header
                    current.update(  # type: ignore[union-attr]
                        time=ts.decode("ascii", "replace"),
                        author=_decode_path(author),
                        email=_decode_path(email),
                        subject=_decode_path(subject),
                    )
                continue
            if pending is not None:
                status, old_sha, new_sha = pending
                current["changes"].append((status, _decode_path(tok), old_sha, new_sha))  # type: ignore[index]
                pending = None
                continue
            text = tok.lstrip(b"\n")
            if text.startswith(b"\x01"):
                if current is not None:
                    yield current
                current = {
                    "sha": text[1:].decode("ascii", "replace"),
                    "time": "",
                    "author": "",
                    "email": "",
                    "subject": "",
                    "changes": [],
                }
                header, header_left = [], 4
            elif text.startswith(b":") and current is not None:
                fields = text[1:].decode("ascii", "replace").split(" ")
                if len(fields) == 5:
                    pending = (fields[4][:1], fields[2], fields[3])
        if current is not None:
            yield current

    def signature_status(self, limit: int) -> list[tuple[str, str]]:
        """(sha, %G? code) for the newest `limit` commits on HEAD. Runs gpg, so callers opt in."""
        out = self.run(["log", f"-n{int(limit)}", "-z", "--format=%H%x1f%G?", "HEAD"])
        rows = []
        for rec in out.split(b"\0"):
            sha, _, code = rec.partition(b"\x1f")
            if sha:
                rows.append(
                    (sha.decode("ascii", "replace"), code.decode("ascii", "replace"))
                )
        return rows

    def count_objects(self) -> dict[str, int]:
        out = self.run(["count-objects", "-v"]).decode("ascii", "replace")
        stats: dict[str, int] = {}
        for line in out.splitlines():
            key, _, value = line.partition(":")
            if value.strip().isdigit():
                stats[key.strip()] = int(value.strip())
        return stats
