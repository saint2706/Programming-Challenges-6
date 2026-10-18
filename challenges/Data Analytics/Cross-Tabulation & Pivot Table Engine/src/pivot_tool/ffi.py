"""Loading the Rust engine through ctypes.

The engine is one shared library with a three-function C ABI (``pt_alloc``, ``pt_call``,
``pt_free``). Everything crosses it as JSON, so this module is small: find the library (building it
with cargo the first time), call it, turn ``{"error": ...}`` into an exception.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pivot_tool.paths import project_root

LIB_ENV = "PIVOT_LIB"


class EngineError(RuntimeError):
    """The engine rejected a request; the message says why."""


class BuildError(RuntimeError):
    """The engine library is missing and could not be built."""


def library_name() -> str:
    if sys.platform == "win32":
        return "pivot_ffi.dll"
    if sys.platform == "darwin":
        return "libpivot_ffi.dylib"
    return "libpivot_ffi.so"


def library_path() -> Path:
    override = os.environ.get(LIB_ENV)
    if override:
        return Path(override)
    return project_root() / "target" / "release" / library_name()


def build_library(wasm: bool = False) -> Path:
    """``cargo build --release`` (or the size-optimised wasm profile). Returns the artifact path."""
    cargo = shutil.which("cargo")
    if cargo is None:
        raise BuildError(
            "cargo is not installed: install Rust (https://rustup.rs) or set PIVOT_LIB"
        )
    root = project_root()
    if wasm:
        cmd = [
            cargo,
            "build",
            "--profile",
            "wasm",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "pivot-ffi",
        ]
        out = root / "target" / "wasm32-unknown-unknown" / "wasm" / "pivot_ffi.wasm"
    else:
        cmd = [cargo, "build", "--release", "-p", "pivot-ffi", "--locked"]
        out = library_path()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BuildError(f"cargo failed:\n{proc.stderr[-2000:]}")
    return out


class Engine:
    """One loaded library. Thread-safe as far as the library is: tables live behind a mutex."""

    def __init__(self, path: Path | None = None, build: bool = True):
        path = path or library_path()
        if not path.exists():
            if not build or os.environ.get(LIB_ENV):
                raise BuildError(f"the engine library is missing: {path}")
            path = build_library()
        self.path = path
        lib = ctypes.CDLL(str(path))
        lib.pt_alloc.argtypes = [ctypes.c_size_t]
        lib.pt_alloc.restype = ctypes.c_void_p
        lib.pt_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.pt_free.restype = None
        lib.pt_call.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.pt_call.restype = ctypes.c_void_p
        self._lib = lib

    def call(self, op: str, payload: dict | None = None, data: bytes = b"") -> Any:
        name = op.encode()
        body = json.dumps(payload if payload is not None else {}).encode()
        out_len = ctypes.c_size_t(0)
        ptr = self._lib.pt_call(
            name, len(name), body, len(body), data, len(data), ctypes.byref(out_len)
        )
        try:
            answer = json.loads(ctypes.string_at(ptr, out_len.value))
        finally:
            self._lib.pt_free(ptr, out_len.value)
        if "error" in answer:
            raise EngineError(answer["error"])
        return answer["ok"]


_default: Engine | None = None


def engine() -> Engine:
    """The process-wide engine, loaded on first use."""
    global _default
    if _default is None:
        _default = Engine()
    return _default
