"""Manim scenes for Kruskal, Prim, and Boruvka, driven by mst.py's Step stream.

Same pattern as the sorting-algorithm visualizer in this repo: the driver
(`MSTScene`) knows nothing algorithm-specific, it only reacts to the
`edge` / `accepted` / `mst_edges` / `total_weight` fields every `*_steps`
generator in mst.py yields. Vertices sit at fixed positions (a circular
layout computed once from the vertex count); an edge is drawn dashed-gray
while under consideration, flashes green and turns solid on acceptance, or
flashes red and disappears on rejection.

    uv run --with manim manim -pql visualize.py KruskalScene
    uv run --with manim manim -pql visualize.py PrimScene
    uv run --with manim manim -pql visualize.py BoruvkaScene
"""

from __future__ import annotations

import math

from manim import *
from mst import boruvka_steps, kruskal_steps, prim_steps
from pacing import (
    CLOSING_HOLD,
    HOLD_SCALE,
    TITLE_MIN_HOLD,
    CaptionPacer,
    motion_time,
    reading_time,
)

DEMO_N = 6
DEMO_EDGES = [
    (0, 1, 4), (0, 2, 4), (1, 2, 2), (1, 3, 5),
    (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6), (4, 5, 3),
]  # fmt: skip

NODE_COLOR = "#3B82C4"
ACCEPT_COLOR = "#3DDC84"
REJECT_COLOR = "#E4572E"
PENDING_COLOR = "#888888"
BG_COLOR = "#101418"

RADIUS = 2.6


def circular_positions(n: int) -> list[tuple[float, float]]:
    if n <= 1:
        return [(0.0, 0.0)] * n
    return [
        (
            RADIUS * math.cos(2 * math.pi * i / n + math.pi / 2),
            RADIUS * math.sin(2 * math.pi * i / n + math.pi / 2),
        )
        for i in range(n)
    ]


class MSTScene(Scene):
    algo_title = "Algorithm"
    algo_fn = None

    def construct(self):
        self.camera.background_color = BG_COLOR
        positions = circular_positions(DEMO_N)

        title = Text(self.algo_title, font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(reading_time(self.algo_title, TITLE_MIN_HOLD) * HOLD_SCALE)

        vertices = []
        for i, (x, y) in enumerate(positions):
            dot = Dot(point=[x, y, 0], radius=0.22, color=NODE_COLOR)
            label = Text(str(i), font_size=20, color=WHITE).move_to([x, y, 0])
            vertices.append(VGroup(dot, label))
        self.play(*[FadeIn(v) for v in vertices])

        weight_text = Text("Total weight: 0", font_size=24, color=WHITE).to_edge(DOWN)
        caption_text = Text("", font_size=22, color=YELLOW).next_to(weight_text, UP)
        self.play(Write(weight_text))

        edge_mobjects: dict[frozenset[int], Line] = {}
        pacer = CaptionPacer()

        for step in self.algo_fn(DEMO_N, DEMO_EDGES):
            if step.edge is None:
                continue
            u, v, _w = step.edge
            key = frozenset((u, v))
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            line = Line([x1, y1, 0], [x2, y2, 0], color=PENDING_COLOR, stroke_width=4)

            new_caption = Text(step.caption, font_size=22, color=YELLOW).next_to(
                weight_text, UP
            )
            motion = motion_time(step)
            self.play(
                Create(line), Transform(caption_text, new_caption), run_time=motion
            )

            if step.accepted:
                accepted_line = Line(
                    [x1, y1, 0], [x2, y2, 0], color=ACCEPT_COLOR, stroke_width=6
                )
                edge_mobjects[key] = accepted_line
                new_weight = Text(
                    f"Total weight: {step.total_weight:g}", font_size=24, color=WHITE
                ).to_edge(DOWN)
                self.play(
                    Transform(line, accepted_line),
                    Transform(weight_text, new_weight),
                    run_time=motion,
                )
                self.wait(pacer.hold_for(step.caption) * HOLD_SCALE)
            else:
                self.play(line.animate.set_color(REJECT_COLOR), run_time=motion)
                self.wait(pacer.hold_for(step.caption) * HOLD_SCALE)
                self.play(FadeOut(line), run_time=0.2)

        self.wait(CLOSING_HOLD * HOLD_SCALE)


class KruskalScene(MSTScene):
    algo_title = "Kruskal: global sorted-edge greedy"
    algo_fn = staticmethod(kruskal_steps)


class PrimScene(MSTScene):
    algo_title = "Prim: one component grows"
    algo_fn = staticmethod(prim_steps)


class BoruvkaScene(MSTScene):
    algo_title = "Boruvka: every component merges in parallel"
    algo_fn = staticmethod(boruvka_steps)
