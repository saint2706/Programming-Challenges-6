"""Equal-area projection for the choropleth: Albers conic, with Alaska/Hawaii/Puerto Rico insets.

A choropleth encodes a value as *colour over area*, so the area the reader sees should be
proportional to the real area.  Web Mercator (the Leaflet default) inflates Alaska to
roughly 3.5x its true size relative to the lower 48, which visually over-weights a few
huge, nearly empty boroughs.  Albers equal-area conic keeps areas honest; the insets
follow the same convention as the USGS/d3 "AlbersUSA" maps.

Units of the projected output are kilometres on a sphere of radius 6371 km.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0

Coord = tuple[float, float]


@dataclass(frozen=True)
class Albers:
    """Spherical Albers equal-area conic (Snyder, *Map Projections: A Working Manual*, eq. 14-1..14-6)."""

    lon0: float
    lat0: float
    lat1: float
    lat2: float

    @property
    def _n(self) -> float:
        return (
            math.sin(math.radians(self.lat1)) + math.sin(math.radians(self.lat2))
        ) / 2.0

    @property
    def _c(self) -> float:
        return math.cos(math.radians(self.lat1)) ** 2 + 2.0 * self._n * math.sin(
            math.radians(self.lat1)
        )

    @property
    def _rho0(self) -> float:
        return (
            math.sqrt(self._c - 2.0 * self._n * math.sin(math.radians(self.lat0)))
            / self._n
        )

    def forward(self, lon: float, lat: float) -> Coord:
        n = self._n
        rho = math.sqrt(max(self._c - 2.0 * n * math.sin(math.radians(lat)), 0.0)) / n
        theta = n * math.radians(
            (lon - self.lon0 + 180.0) % 360.0 - 180.0
        )  # wrap: Aleutians lie past 180
        x = rho * math.sin(theta)
        y = self._rho0 - rho * math.cos(theta)
        return x * EARTH_RADIUS_KM, y * EARTH_RADIUS_KM

    def inverse(self, x: float, y: float) -> Coord:
        n = self._n
        x, y = x / EARTH_RADIUS_KM, y / EARTH_RADIUS_KM
        rho = math.hypot(x, self._rho0 - y)  # n > 0 for every projection used here
        theta = math.atan2(x, self._rho0 - y)
        sin_lat = (self._c - (rho * n) ** 2) / (2.0 * n)
        lat = math.degrees(math.asin(max(-1.0, min(1.0, sin_lat))))
        lon = self.lon0 + math.degrees(theta / n)
        return lon, lat


CONUS = Albers(lon0=-96.0, lat0=37.5, lat1=29.5, lat2=45.5)
ALASKA = Albers(lon0=-154.0, lat0=50.0, lat1=55.0, lat2=65.0)
HAWAII = Albers(lon0=-157.0, lat0=3.0, lat1=8.0, lat2=18.0)
PUERTO_RICO = Albers(lon0=-66.0, lat0=18.0, lat1=8.0, lat2=18.0)

# Alaska is drawn at 35% of its true scale, as in d3's AlbersUSA; it stays equal-area *within*
# itself, but the inset is not comparable in area to the lower 48 (the map says so).
INSET_SCALE = {"AK": 0.35, "HI": 1.0, "PR": 1.0}
_INSET_PROJ = {"AK": ALASKA, "HI": HAWAII, "PR": PUERTO_RICO}
_STATE_GROUP = {"02": "AK", "15": "HI", "72": "PR"}


def group_of(state_fips: str) -> str:
    """Which projection a region belongs to, from its 2-digit state FIPS."""
    return _STATE_GROUP.get(state_fips, "CONUS")


def _walk(geometry: dict):
    """Yield every [x, y] position list (as a mutable reference to its parent) in a Polygon/MultiPolygon."""
    if geometry["type"] == "Polygon":
        polys = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polys = geometry["coordinates"]
    else:
        raise ValueError(f"unsupported geometry type {geometry['type']!r}")
    for poly in polys:
        yield from poly


def _bbox(points: list[Coord]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def project_collection(fc: dict, state_fips_of, decimals: int = 1) -> dict:
    """Return a copy of ``fc`` with Albers-projected coordinates (km), insets placed clear of the lower 48.

    ``state_fips_of(feature) -> str`` names each feature's state so it can be routed to the
    right projection.  Layout is computed from the geometry actually present, so a dataset
    without Alaska simply has no Alaska inset.
    """
    projected: list[dict[str, list[Coord]]] = []
    by_group: dict[str, list[Coord]] = {}
    for feat in fc["features"]:
        group = group_of(state_fips_of(feat))
        proj = CONUS if group == "CONUS" else _INSET_PROJ[group]
        scale = INSET_SCALE.get(group, 1.0)
        rings = []
        for ring in _walk(feat["geometry"]):
            pts = [
                tuple(c * scale for c in proj.forward(lon, lat)) for lon, lat in ring
            ]
            rings.append(pts)
            by_group.setdefault(group, []).extend(pts)
        projected.append({"group": group, "rings": rings})  # type: ignore[dict-item]

    offsets = _layout_insets(by_group)
    out_features = []
    for feat, proj_info in zip(fc["features"], projected, strict=True):
        dx, dy = offsets.get(proj_info["group"], (0.0, 0.0))  # type: ignore[arg-type]
        rebuilt = iter(proj_info["rings"])  # type: ignore[arg-type]
        geom = feat["geometry"]
        polys = (
            [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        )
        new_polys = []
        for poly in polys:
            new_polys.append(
                [
                    [
                        [round(x + dx, decimals), round(y + dy, decimals)]
                        for x, y in next(rebuilt)
                    ]
                    for _ring in poly
                ]
            )
        new_geom = (
            {"type": "Polygon", "coordinates": new_polys[0]}
            if geom["type"] == "Polygon"
            else {"type": "MultiPolygon", "coordinates": new_polys}
        )
        out_features.append(
            {
                "type": "Feature",
                "properties": dict(feat["properties"]),
                "geometry": new_geom,
            }
        )
    return {"type": "FeatureCollection", "features": out_features}


def _layout_insets(by_group: dict[str, list[Coord]]) -> dict[str, Coord]:
    """Translations that park AK/HI/PR in empty space under the lower 48, without overlapping it.

    Alaska and Hawaii sit one ``gap`` below the southernmost lower-48 vertex in the horizontal
    span they occupy, so they tuck in as high as they can; Puerto Rico goes south-east of Florida.
    """
    if "CONUS" not in by_group:
        return {}
    conus = by_group["CONUS"]
    cx0, cy0, cx1, _cy1 = _bbox(conus)
    gap = 0.02 * (cx1 - cx0)

    def floor_over(x_lo: float, x_hi: float) -> float:
        return min(y for x, y in conus if x_lo <= x <= x_hi)

    offsets: dict[str, Coord] = {}
    cursor_x = cx0 + gap
    for group in ("AK", "HI"):
        if group not in by_group:
            continue
        x0, _y0, x1, y1 = _bbox(by_group[group])
        left, right = cursor_x, cursor_x + (x1 - x0)
        offsets[group] = (left - x0, (floor_over(left, right) - gap) - y1)
        cursor_x = right + gap
    if "PR" in by_group:
        # south-east of Florida, level with the lowest point of the lower 48
        x0, y0, x1, _y1 = _bbox(by_group["PR"])
        offsets["PR"] = ((cx1 - gap) - x1, cy0 - y0)
    return offsets


def inset_boxes(
    fc: dict, state_fips_of
) -> dict[str, tuple[float, float, float, float]]:
    """Bounding boxes (x0, y0, x1, y1) of each projection group in an already-projected collection."""
    pts: dict[str, list[Coord]] = {}
    for feat in fc["features"]:
        group = group_of(state_fips_of(feat))
        for ring in _walk(feat["geometry"]):
            pts.setdefault(group, []).extend((x, y) for x, y in ring)
    return {g: _bbox(p) for g, p in pts.items()}
