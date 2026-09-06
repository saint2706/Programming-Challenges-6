"""Manim scene animating the bitmask backtracking search for N-Queens.

Driven directly by `solve_bitmask_steps` in nqueens.py: a queen icon appears
on "place", disappears on "backtrack", and the board flashes green with a
solution counter on "solution". `limit` caps how many complete solutions the
underlying search bothers finding at all, so for demo purposes this animates
a genuine *sample* of the search rather than either (a) exhaustively
rendering every one of a potentially huge solution count, or (b) finding all
of them first and only rendering a slice -- the search itself stops early.

    uv run --with manim manim -pql visualize.py NQueensScene
"""

from __future__ import annotations

from manim import *
from nqueens import solve_bitmask_steps
from pacing import (
    CLOSING_HOLD,
    HOLD_SCALE,
    TITLE_MIN_HOLD,
    CaptionPacer,
    motion_time,
    reading_time,
)

N = 6
SOLUTION_LIMIT = 4

BG_COLOR = "#101418"
LIGHT_SQUARE = "#3A3F4B"
DARK_SQUARE = "#22262E"
QUEEN_COLOR = "#3DDC84"
SOLUTION_FLASH = "#F2C14E"

BOARD_SIZE = 5.0


class NQueensScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        cell = BOARD_SIZE / N

        title_text = f"N-Queens (n={N}): bitmask backtracking"
        title = Text(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(reading_time(title_text, TITLE_MIN_HOLD) * HOLD_SCALE)

        squares = VGroup()
        board_origin_x = -BOARD_SIZE / 2
        board_origin_y = -BOARD_SIZE / 2 + 0.3
        for r in range(N):
            for c in range(N):
                color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
                sq = Square(
                    side_length=cell, fill_color=color, fill_opacity=1, stroke_width=0
                )
                sq.move_to(
                    [
                        board_origin_x + (c + 0.5) * cell,
                        board_origin_y + (r + 0.5) * cell,
                        0,
                    ]
                )
                squares.add(sq)
        self.play(FadeIn(squares))

        counter = Text("Solutions found: 0", font_size=24, color=WHITE).to_edge(DOWN)
        caption = Text("", font_size=22, color=YELLOW).next_to(counter, UP)
        self.play(Write(counter))

        queens: dict[int, Mobject] = {}
        found = 0
        pacer = CaptionPacer()

        def cell_pos(row: int, col: int):
            return [
                board_origin_x + (col + 0.5) * cell,
                board_origin_y + (row + 0.5) * cell,
                0,
            ]

        for step in solve_bitmask_steps(N, limit=SOLUTION_LIMIT):
            new_caption = Text(step.caption, font_size=22, color=YELLOW).next_to(
                counter, UP
            )
            motion = motion_time(step)
            hold = pacer.hold_for(step.caption) * HOLD_SCALE

            if step.action == "place":
                queen = Circle(
                    radius=cell * 0.32,
                    fill_color=QUEEN_COLOR,
                    fill_opacity=1,
                    stroke_width=0,
                )
                queen.move_to(cell_pos(step.row, step.col))
                queens[step.row] = queen
                self.play(
                    FadeIn(queen), Transform(caption, new_caption), run_time=motion
                )
                self.wait(hold)
            elif step.action == "backtrack":
                queen = queens.pop(step.row, None)
                anims = [Transform(caption, new_caption)]
                if queen is not None:
                    anims.append(FadeOut(queen))
                self.play(*anims, run_time=motion)
                self.wait(hold)
            elif step.action == "solution":
                found += 1
                new_counter = Text(
                    f"Solutions found: {found}", font_size=24, color=WHITE
                ).to_edge(DOWN)
                flash = squares.copy().set_fill(SOLUTION_FLASH, opacity=0.25)
                self.play(
                    FadeIn(flash),
                    Transform(counter, new_counter),
                    Transform(caption, new_caption),
                    run_time=motion,
                )
                self.wait(hold)
                self.play(FadeOut(flash), run_time=0.2)

        self.wait(CLOSING_HOLD * HOLD_SCALE)
