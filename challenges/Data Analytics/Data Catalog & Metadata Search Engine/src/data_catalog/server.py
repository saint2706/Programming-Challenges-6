"""Read-only search page over a catalog file. It never opens the cataloged sources."""

from __future__ import annotations

import os
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from data_catalog.store import Catalog

app = FastAPI(title="Data catalog")
# Only answer requests addressed to this machine by name, so a web page cannot read the catalog
# through DNS rebinding. `catalog serve --host X` adds X; DATA_CATALOG_HOSTS adds more.
_extra = [h for h in os.environ.get("DATA_CATALOG_HOSTS", "").split(",") if h]
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", *_extra]
)


def _db() -> str:
    return os.environ.get("DATA_CATALOG_DB", "catalog.sqlite")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        files("data_catalog").joinpath("static/index.html").read_text(encoding="utf-8")
    )


@app.get("/api/search")
def search(
    q: str = Query(..., max_length=200),
    tag: str | None = Query(None, max_length=40),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    with Catalog(_db()) as cat:
        return cat.search(q, limit=limit, tag=tag)


@app.get("/api/dataset")
def dataset(key: str = Query(..., max_length=2000)) -> dict:
    with Catalog(_db()) as cat:
        d = cat.get(key)
    if d is None:
        raise HTTPException(404, "no such dataset")
    return d
