"""The scorecard as an OpenVINO graph, so batch scoring can run on the CPU or the iGPU.

A scorecard is lookups and a sum, which maps onto tensor ops without any trained weights:

* bin index  = number of cut points strictly below x      (``Greater`` + ``ReduceSum``)
* special    = ``x`` equals a sentinel code               (``Equal`` + ``ReduceLogicalOr``)
* missing    = ``x != x``, which is only true for NaN     (``NotEqual``)
* points     = ``Gather`` from the feature's points table, then ``Add`` across features

Inputs are float32, so a value within one float32 step of a cut point can land in the neighbouring
bin; ``parity`` measures how often against the exact float64 NumPy score. The GPU plugin defaults to
float16, which moves many values across cut points (0.09% of rows, off by up to 15 points), so the
model is compiled with ``INFERENCE_PRECISION_HINT=f32``. The NPU is not supported: it is float16-only
and needs static shapes. The gradient-boosting benchmark has no graph (no tree-ensemble op).
"""

from __future__ import annotations

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import polars as pl

from credit_scorecard.scorecard import Scorecard


def build_model(card: Scorecard) -> ov.Model:
    """Graph from float32 ``[N, n_features]`` (column order = ``card.features``) to float32 ``[N]``."""
    x = ops.parameter([-1, len(card.features)], np.float32, name="x")
    axis1 = ops.constant(np.array([1], np.int64))
    total = None
    for j, f in enumerate(card.features):
        binning = card.binnings[f]
        col = ops.gather(x, ops.constant(np.int64(j)), ops.constant(np.int64(1)))  # [N]
        col2 = ops.unsqueeze(col, ops.constant(np.array([1], np.int64)))  # [N, 1]
        edges = ops.constant(np.array(binning.edges, np.float32).reshape(1, -1))
        n_bins = len(binning.bins)
        index = (
            ops.reduce_sum(
                ops.convert(ops.greater(col2, edges), "i32"), axis1, keep_dims=False
            )
            if binning.edges
            else ops.broadcast(ops.constant(np.int32(0)), ops.shape_of(col))
        )
        special_slot, missing_slot = binning._slots()
        if binning.specials:
            sentinels = ops.constant(
                np.array(binning.specials, np.float32).reshape(1, -1)
            )
            is_special = ops.reduce_logical_or(
                ops.equal(col2, sentinels), axis1, keep_dims=False
            )
            index = ops.select(is_special, ops.constant(np.int32(special_slot)), index)
        index = ops.select(
            ops.not_equal(col, col), ops.constant(np.int32(missing_slot)), index
        )
        table = np.array(card.points[f], np.float32)
        assert len(table) == n_bins + 1
        points = ops.gather(ops.constant(table), index, ops.constant(np.int64(0)))
        total = points if total is None else ops.add(total, points)
    return ov.Model([ops.result(total, name="score")], [x], "scorecard")


class OvScorer:
    """Compiled scorecard. ``device`` is "CPU", "GPU" (the iGPU) or "AUTO"."""

    def __init__(self, card: Scorecard, device: str = "AUTO") -> None:
        self.card = card
        self.device = device
        self._compiled = ov.Core().compile_model(
            build_model(card), device, {"INFERENCE_PRECISION_HINT": "f32"}
        )

    def matrix(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.card.features).to_numpy().astype(np.float32)

    def score(self, df: pl.DataFrame | np.ndarray) -> np.ndarray:
        x = self.matrix(df) if isinstance(df, pl.DataFrame) else df
        return np.rint(self._compiled(x)[0]).astype(np.int64)


def available_devices() -> dict[str, str]:
    core = ov.Core()
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


def parity(card: Scorecard, df: pl.DataFrame, device: str) -> float:
    """Fraction of rows where the OpenVINO score equals the exact NumPy score."""
    return float(np.mean(OvScorer(card, device).score(df) == card.score(df)))
