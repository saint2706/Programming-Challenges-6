"""A LightGBM hedonic model and the three things the comp finder reads out of it.

* ``predict``: the pure-AVM reference (time-adjusted log price, no comps).
* ``shap_weight``: mean |SHAP| per feature, the "what moves price" weights for a weighted distance.
* ``leaves``: for every sale, the leaf it lands in for each tree. Two sales that share leaves are
  treated alike by the model; the weighted fraction of shared leaves is the tree-ensemble
  proximity (the random-forest proximity / GAP idea, applied to boosted trees; see the README).

``path_share`` records which features each leaf's decision path splits on, so a proximity can be
explained: every shared leaf passes its tree's weight to the features on its path.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from scipy import sparse

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}
N_TREES = 300


def _walk(
    node: dict, path: list[int], paths: dict[int, list[int]], values: dict[int, float]
):
    if "leaf_index" in node:
        paths[node["leaf_index"]] = path
        values[node["leaf_index"]] = node["leaf_value"]
        return
    below = [*path, node["split_feature"]]
    _walk(node["left_child"], below, paths, values)
    _walk(node["right_child"], below, paths, values)


@dataclass
class HedonicGBM:
    booster: lgb.Booster
    n_features: int
    num_leaves: int
    tree_weight: (
        np.ndarray
    )  # (T,) share of the model's movement each tree carries, sums to 1
    path_share: sparse.csr_matrix  # (T*num_leaves, d); each non-empty row sums to 1
    shap_weight: np.ndarray  # (d,)

    @classmethod
    def fit(
        cls,
        z: np.ndarray,
        y: np.ndarray,
        n_trees: int = N_TREES,
        params: dict | None = None,
    ) -> HedonicGBM:
        params = {**PARAMS, **(params or {})}
        booster = lgb.train(params, lgb.Dataset(z, y), num_boost_round=n_trees)
        d, nl = z.shape[1], params["num_leaves"]
        info = booster.dump_model()["tree_info"]
        rows, cols, vals = [], [], []
        leaf_value = np.zeros((len(info), nl))
        for t, tree in enumerate(info):
            paths: dict[int, list[int]] = {}
            values: dict[int, float] = {}
            _walk(tree["tree_structure"], [], paths, values)
            for leaf, path in paths.items():
                leaf_value[t, leaf] = values[leaf]
                for f in path:
                    rows.append(t * nl + leaf)
                    cols.append(f)
                    vals.append(1.0 / len(path))
        share = sparse.csr_matrix((vals, (rows, cols)), shape=(len(info) * nl, d))
        leaf_idx = booster.predict(z, pred_leaf=True)
        movement = np.array(
            [leaf_value[t, leaf_idx[:, t]].std() for t in range(len(info))]
        )
        weight = movement / movement.sum()
        shap = np.abs(booster.predict(z, pred_contrib=True)[:, :d]).mean(axis=0)
        return cls(booster, d, nl, weight, share, shap)

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.booster.predict(z)

    def leaves(self, z: np.ndarray) -> sparse.csr_matrix:
        """One-hot of (tree, leaf) per row: ``(n, T*num_leaves)`` with T ones per row."""
        idx = self.booster.predict(z, pred_leaf=True)
        n, t = idx.shape
        cols = (np.arange(t)[None, :] * self.num_leaves + idx).ravel()
        return sparse.csr_matrix(
            (np.ones(n * t), cols, np.arange(0, n * t + 1, t)),
            shape=(n, t * self.num_leaves),
        )

    def column_weight(self) -> np.ndarray:
        """Each (tree, leaf) column carries its tree's weight."""
        return np.repeat(self.tree_weight, self.num_leaves)
