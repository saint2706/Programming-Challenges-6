"""Albers equal-area projection and the Alaska/Hawaii/Puerto Rico inset layout."""

from __future__ import annotations

import json
import math

import pytest
from geo_heatmap import DATA
from projection import (
    CONUS,
    EARTH_RADIUS_KM,
    Albers,
    group_of,
    inset_boxes,
    project_collection,
)


def ring_area(ring) -> float:
    pts = [tuple(p) for p in ring]
    return (
        abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True)
            )
        )
        / 2
    )


def feature_area(feature) -> float:
    geom = feature["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return sum(ring_area(p[0]) - sum(ring_area(h) for h in p[1:]) for p in polys)


@pytest.fixture(scope="module")
def states():
    return json.loads((DATA / "us_states_2017_20m.geojson").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def projected(states):
    return project_collection(states, lambda f: f["properties"]["fips"])


class TestAlbers:
    def test_origin_maps_to_zero(self):
        x, y = CONUS.forward(CONUS.lon0, CONUS.lat0)
        assert x == pytest.approx(0, abs=1e-9) and y == pytest.approx(0, abs=1e-9)

    @pytest.mark.parametrize(
        "lon,lat", [(-122.4, 37.8), (-80.2, 25.8), (-70.0, 44.0), (-104.0, 31.0)]
    )
    def test_round_trip(self, lon, lat):
        assert CONUS.inverse(*CONUS.forward(lon, lat)) == pytest.approx(
            (lon, lat), abs=1e-9
        )

    @pytest.mark.parametrize("lat", [26.0, 37.0, 48.0])
    def test_equal_area_against_spherical_cell_area(self, lat):
        """A 0.1 x 0.1 degree cell must keep its true area R^2 * dlon * (sin(lat2) - sin(lat1))."""
        lon, d = -100.0, 0.1
        corners = [(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d)]
        projected_area = ring_area([CONUS.forward(*c) for c in corners])
        true_area = (
            EARTH_RADIUS_KM**2
            * math.radians(d)
            * (math.sin(math.radians(lat + d)) - math.sin(math.radians(lat)))
        )
        assert projected_area == pytest.approx(true_area, rel=2e-3)

    def test_antimeridian_is_wrapped(self):
        alaska = Albers(-154.0, 50.0, 55.0, 65.0)
        west, east = alaska.forward(179.9, 52.0), alaska.forward(-179.9, 52.0)
        assert math.dist(west, east) < 50  # 0.2 degrees apart, not 360 degrees apart


class TestLayout:
    def test_state_areas_are_preserved(self, projected):
        """Texas and Montana are in the lower-48 projection: areas must match reality (total area, 1:20M shapes)."""
        area = {f["properties"]["abbr"]: feature_area(f) for f in projected["features"]}
        assert area["TX"] == pytest.approx(695_662, rel=0.015)
        assert area["MT"] == pytest.approx(380_831, rel=0.015)

    def test_insets_are_scaled_as_documented(self, states, projected):
        raw = {f["properties"]["abbr"]: f for f in projected["features"]}
        assert feature_area(raw["AK"]) > 0
        # Alaska is drawn at 0.35 linear scale: its true area is ~1.72M km2, so ~0.21M km2 on the page
        assert feature_area(raw["AK"]) == pytest.approx(1_723_337 * 0.35**2, rel=0.15)

    def test_insets_do_not_overlap_each_other_or_the_lower_48(self, projected):
        fips = lambda f: f["properties"]["fips"]
        boxes = inset_boxes(projected, fips)
        assert set(boxes) == {"CONUS", "AK", "HI", "PR"}
        names = ["AK", "HI", "PR"]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ax0, ay0, ax1, ay1 = boxes[a]
                bx0, by0, bx1, by1 = boxes[b]
                assert ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0, (
                    f"{a} overlaps {b}"
                )
        conus_points = [
            (x, y)
            for f in projected["features"]
            if group_of(f["properties"]["fips"]) == "CONUS"
            for poly in f["geometry"]["coordinates"]
            for ring in poly
            for x, y in ring
        ]
        for name in names:
            x0, y0, x1, y1 = boxes[name]
            assert not any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in conus_points), (
                f"{name} sits on the lower 48"
            )

    def test_dataset_without_alaska_has_no_alaska_inset(self, states):
        subset = {
            "type": "FeatureCollection",
            "features": [
                f
                for f in states["features"]
                if f["properties"]["abbr"] not in ("AK", "HI", "PR")
            ],
        }
        boxes = inset_boxes(
            project_collection(subset, lambda f: f["properties"]["fips"]),
            lambda f: f["properties"]["fips"],
        )
        assert set(boxes) == {"CONUS"}

    def test_projection_does_not_mutate_input(self, states):
        before = json.dumps(states["features"][0])
        project_collection(states, lambda f: f["properties"]["fips"])
        assert json.dumps(states["features"][0]) == before

    def test_properties_and_order_survive(self, states, projected):
        assert [f["properties"] for f in projected["features"]] == [
            f["properties"] for f in states["features"]
        ]

    def test_unsupported_geometry(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                }
            ],
        }
        with pytest.raises(ValueError, match="unsupported"):
            project_collection(fc, lambda f: "01")

    def test_group_of(self):
        assert [group_of(s) for s in ("02", "15", "72", "48")] == [
            "AK",
            "HI",
            "PR",
            "CONUS",
        ]
