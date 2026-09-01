"""Manim scenes for the 14 sorting algorithms in sorting_algorithms.py.

The animation driver (SortRaceScene) is entirely generic: it reacts to the
``compare`` / ``swap`` / ``write`` / ``sorted_idx`` / ``aux`` fields of each
``Step`` and knows nothing about any specific algorithm. Bars stay at a fixed
x-position per array index (a swap is shown as two bars simultaneously
changing height and flashing red, not as bars physically sliding past each
other) -- this keeps 14 correct, watchable animations from one small driver
instead of 14 bespoke ones.

Render one scene, low quality, to iterate fast:
    manim -pql visualize.py BubbleSortScene

Render everything at once:
    manim -qm visualize.py SelectionSortScene BubbleSortScene InsertionSortScene \
        MergeSortScene QuickSortScene HeapSortScene CycleSortScene \
        ThreeWayMergeSortScene CountingSortScene RadixSortScene BucketSortScene \
        PigeonholeSortScene IntroSortScene TimSortScene
"""

from manim import *

from sorting_algorithms import ALGORITHMS, BASE_ARRAY

BG_COLOR = "#101418"
DEFAULT_COLOR = "#3B82C4"
COMPARE_COLOR = YELLOW
SWAP_COLOR = "#E4572E"
WRITE_COLOR = "#3DDC84"
SORTED_COLOR = "#2E8B57"
EMPTY_COLOR = GRAY_D

BAR_WIDTH = 0.9
SPACING = 1.15
HEIGHT_SCALE = 0.12
BASELINE_Y = -2.0
INDEX_LABEL_Y = BASELINE_Y - 0.35
RUNS_BAND_Y = BASELINE_Y - 0.65
CAPTION_Y = -2.85
AUX_ANCHOR_Y = -3.5
ABOVE_BARS_Y = 2.4


class SortRaceScene(Scene):
    algo_title = "Algorithm"
    algo_key = None
    subtitle = ""

    def bar_x(self, i, n):
        return (i - (n - 1) / 2) * SPACING

    def bar_geometry(self, i, n, value, color):
        x = self.bar_x(i, n)
        if value is None:
            h = 0.15
            rect = Rectangle(width=BAR_WIDTH, height=h, fill_color=EMPTY_COLOR,
                              fill_opacity=0.25, stroke_color=EMPTY_COLOR, stroke_width=1)
            rect.move_to([x, BASELINE_Y + h / 2, 0])
            label = Text("", font_size=22).move_to([x, BASELINE_Y + h + 0.25, 0])
            return rect, label
        h = max(value * HEIGHT_SCALE, 0.15)
        rect = Rectangle(width=BAR_WIDTH, height=h, fill_color=color, fill_opacity=0.9,
                          stroke_color=WHITE, stroke_width=1)
        rect.move_to([x, BASELINE_Y + h / 2, 0])
        label = Text(str(value), font_size=22, color=WHITE).move_to([x, BASELINE_Y + h + 0.28, 0])
        return rect, label

    def classify_color(self, i, step):
        if i in step.sorted_idx:
            return SORTED_COLOR
        if step.compare and i in step.compare:
            return COMPARE_COLOR
        if step.swap and i in step.swap:
            return SWAP_COLOR
        if step.write and i in step.write:
            return WRITE_COLOR
        return DEFAULT_COLOR

    def construct(self):
        self.camera.background_color = BG_COLOR
        n = len(BASE_ARRAY)

        title = Text(self.algo_title, font_size=40, color=WHITE).to_edge(UP, buff=0.4)
        self.add(title)
        if self.subtitle:
            sub = Text(self.subtitle, font_size=20, color=GRAY_B).next_to(title, DOWN, buff=0.15)
            self.add(sub)

        idx_labels = VGroup(*[
            Text(str(i), font_size=18, color=GRAY_B).move_to([self.bar_x(i, n), INDEX_LABEL_Y, 0])
            for i in range(n)
        ])
        self.add(idx_labels)

        self.bars, self.labels = [], []
        for i, v in enumerate(BASE_ARRAY):
            rect, label = self.bar_geometry(i, n, v, DEFAULT_COLOR)
            self.bars.append(rect)
            self.labels.append(label)
        self.add(*self.bars, *self.labels)

        # A fixed anchor point, not to_edge() on an empty string -- an empty
        # Text has no glyphs, so to_edge() on it is a no-op and it silently
        # sits near the origin instead of at the intended screen position.
        self.caption = Text(" ", font_size=22, color=WHITE).move_to([0, CAPTION_Y, 0])
        self.add(self.caption)

        self.aux = VGroup()
        self.aux_anchor = np.array([0.0, AUX_ANCHOR_Y, 0.0])
        self.add(self.aux)
        self._last_aux_data = None

        fn = ALGORITHMS[self.algo_key]
        arr = list(BASE_ARRAY)
        for step in fn(arr):
            self.play_step(step, n)

        self.play(FadeOut(self.aux))
        self.wait(1.2)

    def play_step(self, step, n):
        anims = []
        for i in range(n):
            new_v = step.array[i] if i < len(step.array) else None
            color = self.classify_color(i, step)
            new_rect, new_label = self.bar_geometry(i, n, new_v, color)
            anims.append(Transform(self.bars[i], new_rect))
            anims.append(Transform(self.labels[i], new_label))

        new_caption = Text(step.caption, font_size=22, color=WHITE).move_to([0, CAPTION_Y, 0])
        anims.append(Transform(self.caption, new_caption))

        run_time = 0.25 if (step.swap or step.write) else (0.15 if step.compare else 0.2)

        pending_swap = None
        if step.aux is not None and step.aux != self._last_aux_data:
            old_aux = self.aux
            new_aux = self.build_aux(step.aux, n)
            anims.append(FadeOut(old_aux, run_time=run_time))
            anims.append(FadeIn(new_aux, run_time=run_time))
            pending_swap = (old_aux, new_aux)
            self._last_aux_data = step.aux

        self.play(*anims, run_time=run_time)

        if pending_swap:
            old_aux, new_aux = pending_swap
            self.remove(old_aux)
            self.aux = new_aux

    # -- auxiliary panel dispatch -------------------------------------------------

    def build_aux(self, aux, n):
        t = aux.get("type")
        if t == "segment":
            return self._aux_segment(aux, n)
        if t == "pivot":
            return self._aux_pivot(aux, n)
        if t == "heap":
            return self._aux_heap(aux, n)
        if t == "cycle":
            return self._aux_cycle(aux, n)
        if t == "mode":
            return self._aux_mode(aux)
        if t == "runs":
            return self._aux_runs(aux, n)
        if t in ("radix", "buckets", "holes", "count"):
            return self._aux_columns(aux)
        return VGroup().move_to(self.aux_anchor)

    def _aux_segment(self, aux, n):
        l, r = aux["range"]
        y = ABOVE_BARS_Y
        x1 = self.bar_x(l, n) - BAR_WIDTH / 2
        x2 = self.bar_x(r - 1, n) + BAR_WIDTH / 2
        line = Line([x1, y, 0], [x2, y, 0], color=YELLOW)
        ticks = VGroup(
            Line([x1, y - 0.1, 0], [x1, y + 0.1, 0], color=YELLOW),
            Line([x2, y - 0.1, 0], [x2, y + 0.1, 0], color=YELLOW),
        )
        label = Text(f"[{l}, {r})", font_size=18, color=YELLOW).next_to(line, UP, buff=0.1)
        return VGroup(line, ticks, label)

    def _aux_pivot(self, aux, n):
        x = self.bar_x(aux["idx"], n)
        marker = Triangle(color=ORANGE, fill_opacity=1).scale(0.15).rotate(PI).move_to([x, ABOVE_BARS_Y, 0])
        label = Text("pivot", font_size=16, color=ORANGE).next_to(marker, UP, buff=0.05)
        return VGroup(marker, label)

    def _aux_heap(self, aux, n):
        size = aux["size"]
        y = ABOVE_BARS_Y
        dots = VGroup(*[Dot([self.bar_x(i, n), y, 0], radius=0.05, color=PURPLE_B) for i in range(size)])
        lines = VGroup()
        for i in range(size):
            for child in (2 * i + 1, 2 * i + 2):
                if child < size:
                    lines.add(Line([self.bar_x(i, n), y, 0], [self.bar_x(child, n), y, 0],
                                    color=PURPLE_B, stroke_width=1.5))
        return VGroup(lines, dots)

    def _aux_cycle(self, aux, n):
        x = self.bar_x(aux["start"], n)
        marker = Circle(radius=0.15, color=TEAL, fill_opacity=0.6).move_to([x, ABOVE_BARS_Y, 0])
        label = Text("cycle start", font_size=16, color=TEAL).next_to(marker, UP, buff=0.05)
        return VGroup(marker, label)

    def _aux_mode(self, aux):
        mode = aux["mode"]
        colors = {"quick": "#3B82C4", "heap": PURPLE_B, "insertion": ORANGE}
        label = Text(f"Mode: {mode.upper()}", font_size=22, color=colors.get(mode, WHITE))
        label.to_corner(UR, buff=0.5)
        return VGroup(label)

    def _aux_runs(self, aux, n):
        palette = ["#3DDC84", "#F2C14E", "#E4572E", "#8E7DBE", "#3B82C4"]
        group = VGroup()
        for k, (l, r) in enumerate(aux["runs"]):
            x1 = self.bar_x(l, n) - BAR_WIDTH / 2
            x2 = self.bar_x(r - 1, n) + BAR_WIDTH / 2
            band = Rectangle(width=(x2 - x1), height=0.12, fill_color=palette[k % len(palette)],
                              fill_opacity=0.9, stroke_width=0)
            band.move_to([(x1 + x2) / 2, RUNS_BAND_Y, 0])
            group.add(band)
        return group

    def _aux_columns(self, aux):
        # No heading here on purpose: the caption line above already states
        # what's happening ("Scatter 29 into bucket 4") -- a duplicate
        # heading would eat into the tight vertical budget below the bars.
        t = aux["type"]
        if t == "radix":
            groups, labels = aux["buckets"], [str(i) for i in range(10)]
        elif t == "buckets":
            groups = aux["buckets"]
            labels = [f"B{i}" for i in range(len(groups))]
        elif t == "holes":
            lo, holes = aux["lo"], aux["holes"]
            occupied = [(lo + i, h) for i, h in enumerate(holes) if h]
            if not occupied:
                return VGroup().move_to(self.aux_anchor)
            labels = [str(v) for v, _ in occupied]
            groups = [h for _, h in occupied]
        else:  # count
            lo, table = aux["lo"], aux["table"]
            labels = [str(lo + i) for i in range(len(table))]
            groups = [[v] if v else [] for v in table]

        cols = VGroup()
        for label, g in zip(labels, groups):
            content = "\n".join(str(v) for v in g) if g else "-"
            col_text = Text(content, font_size=16, color=WHITE)
            idx_text = Text(label, font_size=14, color=GRAY_B)
            cols.add(VGroup(col_text, idx_text).arrange(DOWN, buff=0.08))
        row = cols.arrange(RIGHT, buff=0.28)
        if row.width > 12.5:
            row.scale(12.5 / row.width)
        row.move_to(self.aux_anchor)
        return row


class SelectionSortScene(SortRaceScene):
    algo_title = algo_key = "Selection Sort"
    subtitle = "Unstable | O(n^2) always | minimal writes"


class BubbleSortScene(SortRaceScene):
    algo_title = algo_key = "Bubble Sort"
    subtitle = "Stable | O(n^2) | early-exit when no swaps happen"


class InsertionSortScene(SortRaceScene):
    algo_title = algo_key = "Insertion Sort"
    subtitle = "Stable | O(n^2) worst | O(n) best on nearly-sorted input"


class MergeSortScene(SortRaceScene):
    algo_title = algo_key = "Merge Sort"
    subtitle = "Stable | O(n log n) always | needs O(n) extra space"


class QuickSortScene(SortRaceScene):
    algo_title = algo_key = "Quick Sort"
    subtitle = "Unstable | O(n log n) average | O(n^2) worst case"


class HeapSortScene(SortRaceScene):
    algo_title = algo_key = "Heap Sort"
    subtitle = "Unstable | O(n log n) always | O(1) extra space"


class CycleSortScene(SortRaceScene):
    algo_title = algo_key = "Cycle Sort"
    subtitle = "Stable-ish | O(n^2) | minimizes total number of writes"


class ThreeWayMergeSortScene(SortRaceScene):
    algo_title = algo_key = "3-Way Merge Sort"
    subtitle = "Stable | O(n log3 n) comparisons | fewer merge levels than 2-way"


class CountingSortScene(SortRaceScene):
    algo_title = algo_key = "Counting Sort"
    subtitle = "Stable | O(n + k), k = value range | non-comparison"


class RadixSortScene(SortRaceScene):
    algo_title = algo_key = "Radix Sort"
    subtitle = "Stable | O(d * (n + b)), d = digits | non-comparison"


class BucketSortScene(SortRaceScene):
    algo_title = algo_key = "Bucket Sort"
    subtitle = "Stable | O(n + k) average | needs a roughly uniform distribution"


class PigeonholeSortScene(SortRaceScene):
    algo_title = algo_key = "Pigeonhole Sort"
    subtitle = "Stable | O(n + range) | mechanically close to Counting Sort"


class IntroSortScene(SortRaceScene):
    algo_title = algo_key = "IntroSort"
    subtitle = "Unstable | O(n log n) worst-case guaranteed | Quick -> Heap -> Insertion"


class TimSortScene(SortRaceScene):
    algo_title = algo_key = "TimSort"
    subtitle = "Stable | O(n log n) worst, O(n) best | runs + merge, powers Python's sort"
