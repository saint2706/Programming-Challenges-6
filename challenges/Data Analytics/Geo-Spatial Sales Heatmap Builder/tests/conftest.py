"""Shared fixtures: a tiny hand-made map so most tests never touch the bundled Census files."""

from __future__ import annotations

import json

import pytest


def square(x: float, y: float, size: float = 1.0) -> dict:
    """A unit square as a GeoJSON Polygon geometry (counter-clockwise, closed ring)."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
        ],
    }


def grid_features(nx: int, ny: int, name_of=None) -> list[dict]:
    """``nx * ny`` unit squares sharing exact edges; key ``C{col}{row}``, row-major from the bottom left."""
    feats = []
    for row in range(ny):
        for col in range(nx):
            key = f"C{col}{row}"
            name = name_of(key) if name_of else f"Cell {col},{row}"
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"id": key, "name": name},
                    "geometry": square(col, row),
                }
            )
    return feats


@pytest.fixture
def grid_geojson(tmp_path):
    """A 3x2 grid written to disk; returns (path, features)."""
    feats = grid_features(3, 2)
    path = tmp_path / "grid.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    return path, feats
