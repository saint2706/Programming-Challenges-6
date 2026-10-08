"""A labeling project on disk: SQLite store + embeddings (+ optional eval set), and the model around them."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import pairwise
from math import isnan
from pathlib import Path

import numpy as np

from active_labeling import explain, stats, strategies
from active_labeling.model import Head
from active_labeling.store import Store

# ``spacing``: the benchmark measured prediction change between rounds of 50 labels, so the signal is
# computed between rounds at least this many labels apart (one-label rounds change almost nothing)
DEFAULT_RULE = {"threshold": 0.02, "k": 3, "spacing": 50}


class Project:
    def __init__(self, path):
        self.dir = Path(path)
        db = self.dir / "project.db"
        if not db.exists():
            raise FileNotFoundError(
                f"{self.dir} is not a project; create one with `uv run active-labeling init {self.dir} --demo`"
            )
        self.store = Store(db)
        self.X = np.load(self.dir / "embeddings.npy").astype(np.float64)
        self.classes = self.store.classes()
        self.class_id = {c: i for i, c in enumerate(self.classes)}
        self.texts = self.store.texts()
        self.C = float(self.store.get("C"))
        self.rule = {**DEFAULT_RULE, **json.loads(self.store.get("rule"))}
        self.has_gold = any(g is not None for g in self.store.gold())
        self.X_eval = self.y_eval = None
        eval_path = self.dir / "eval.npz"
        if eval_path.exists():
            with np.load(eval_path) as z:
                self.X_eval, self.y_eval = z["X"].astype(np.float64), z["y"]

    @classmethod
    def create(
        cls,
        path,
        texts,
        X,
        classes,
        *,
        gold=None,
        C=10.0,
        rule=None,
        X_eval=None,
        y_eval=None,
    ):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        np.save(path / "embeddings.npy", np.asarray(X, dtype=np.float32))
        store = Store(path / "project.db")
        store.set_classes(classes)
        store.add_items(texts, gold)
        store.set("C", repr(float(C)))
        store.set("rule", json.dumps(rule or DEFAULT_RULE))
        if X_eval is not None:
            np.savez(
                path / "eval.npz",
                X=np.asarray(X_eval, dtype=np.float32),
                y=np.asarray(y_eval),
            )
        return cls(path)

    def labeled(self):
        current = self.store.current_labels()
        idx = np.array(sorted(current), dtype=np.int64)
        y = np.array([self.class_id[current[i]] for i in idx], dtype=np.int64)
        return idx, y

    def _head(self, idx, y):
        return Head(len(self.classes), self.C).fit(self.X[idx], y)

    def state(self) -> strategies.State:
        idx, y = self.labeled()
        K = len(self.classes)
        P = (
            self._head(idx, y).proba(self.X)
            if len(idx)
            else np.full((len(self.X), K), 1.0 / K)
        )
        return strategies.State(self.X, idx, y, P, K, self.C)

    def _clusters(self, n_clusters: int = 150) -> np.ndarray:
        path = self.dir / f"clusters-{n_clusters}.npy"
        if path.exists():
            return np.load(path)
        clusters = strategies.cluster_pool(self.X, n_clusters)
        np.save(path, clusters)
        return clusters

    def suggest(self, name: str, b: int, seed: int = 0):
        if b < 1:
            raise ValueError("batch size must be at least 1")
        if name not in strategies.NAMES:
            raise ValueError(
                f"unknown strategy {name!r}; choose from {strategies.NAMES}"
            )
        state = self.state()
        clusters = self._clusters() if name == "cluster-margin" else None
        strategy = strategies.make(name, clusters=clusters)
        selection = strategy.select(state, b, np.random.default_rng(seed))
        reasons = explain.explain(state, selection, clusters=clusters)
        return selection, reasons, state

    def submit(self, labels: dict[int, str]):
        """Write a batch of labels (all or nothing), then record a round if the model input changed."""
        known = set(self.classes)
        for item, name in labels.items():
            if not 0 <= item < len(self.texts):
                raise ValueError(f"no item {item}")
            if name not in known:
                raise ValueError(f"{name!r} is not one of the {len(known)} classes")
        for item, name in labels.items():
            self.store.label(int(item), name)
        return self.refresh()

    def refresh(self):
        idx, y = self.labeled()
        if len(idx) == 0:
            return None
        current = self.store.current_labels()
        sig = hashlib.sha256(json.dumps(sorted(current.items())).encode()).hexdigest()[
            :16
        ]
        rounds = self.store.rounds()
        last = rounds[-1] if rounds else None
        if last is not None and last["sig"] == sig:
            return None
        head = self._head(idx, y)
        pred = head.predict(self.X).astype(np.int16)
        change = None if last is None else float((pred != last["pred"]).mean())
        accuracy = None
        if self.X_eval is not None:
            accuracy = float((head.predict(self.X_eval) == self.y_eval).mean())
        n_classes = len(np.unique(y))
        self.store.add_round(len(idx), n_classes, change, accuracy, pred, sig)
        return {"n_labeled": len(idx), "change": change, "accuracy": accuracy}

    def simulate(self, ids):
        gold = self.store.gold()
        if not self.has_gold:
            raise ValueError(
                "this project has no gold labels to simulate an annotator with"
            )
        missing = [i for i in ids if gold[i] is None]
        if missing:
            raise ValueError(f"items {missing[:5]} have no gold label")
        return self.submit({int(i): gold[i] for i in ids})

    def status(self) -> dict:
        idx, y = self.labeled()
        rounds = self.store.rounds()
        signal = rounds[:1]
        for r in rounds[1:]:
            if r["n_labeled"] - signal[-1]["n_labeled"] >= self.rule["spacing"]:
                signal.append(r)
        change = [float("nan")] * min(1, len(signal)) + [
            float((b["pred"] != a["pred"]).mean()) for a, b in pairwise(signal)
        ]
        at = stats.stop_round(change, self.rule["threshold"], self.rule["k"])
        return {
            "items": len(self.texts),
            "labeled": len(idx),
            "classes": len(self.classes),
            "classes_with_label": len(np.unique(y)),
            "counts": np.bincount(y, minlength=len(self.classes)).tolist(),
            "rounds": [
                {
                    "n_labeled": r["n_labeled"],
                    "change": r["change"],
                    "accuracy": r["accuracy"],
                }
                for r in rounds
            ],
            "rule": self.rule,
            "signal_rounds": [
                {"n_labeled": r["n_labeled"], "change": None if isnan(c) else c}
                for r, c in zip(signal, change, strict=True)
            ],
            "stop": {
                "fired": at is not None,
                "at_labels": None if at is None else signal[at]["n_labeled"],
            },
        }

    def export(self, path) -> None:
        current, gold = self.store.current_labels(), self.store.gold()
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "text", "label", "gold"])
            for i, text in enumerate(self.texts):
                w.writerow([i, text, current.get(i, ""), gold[i] or ""])
