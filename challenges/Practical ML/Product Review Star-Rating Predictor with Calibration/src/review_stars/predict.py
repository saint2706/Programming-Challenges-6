"""Score one review with a finished run: embed, ensemble the MLP heads, calibrate, conformalize.

Everything is read from ``results/`` as plain files: head weights (``torch.load`` with
``weights_only=True``), calibrators and the conformal threshold as arrays and JSON. Nothing is
unpickled. The headline model per framing is the 5-seed MLP ensemble with temperature scaling and
randomized APS sets at the benchmark's ``alpha``.

The uniform draw that randomized APS needs is a hash of the review text, so the same review always
gets the same set (and the app does not flicker on a rerun).
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from review_stars import calibrate, conformal, embed, heads, pipeline
from review_stars import text as text_mod
from review_stars.config import FRAMINGS, results_dir
from review_stars.probs import Predictions, expected_star

MISSING = "no finished benchmark in {path}; run `review-stars benchmark` first"


@dataclass
class Bundle:
    members: dict[str, list[heads.Head]]
    calibrators: dict[str, calibrate.Temperature]
    qhat: dict[str, float]
    alpha: float
    text_mode: str
    models: dict[str, str]  # framing -> the model name used ("ens-..." or "mlp-...")


def _read(results: Path, stage: str):
    js, npz = pipeline._paths(results, stage)
    if not js.exists():
        raise FileNotFoundError(MISSING.format(path=results))
    stored = json.loads(js.read_text(encoding="utf-8"))
    arrays = None
    if stored.get("has_arrays"):
        with np.load(npz, allow_pickle=False) as z:
            arrays = dict(z)
    return stored["payload"], arrays


def load_bundle(results=None) -> Bundle:
    results = Path(results) if results is not None else results_dir()
    report_path = results / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(MISSING.format(path=results))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = report["config"]
    _, cal_arrays = _read(results, "calibrate")
    conf_payload, _ = _read(results, "conformal")
    out = Bundle({}, {}, {}, float(conf_payload["alpha"]), cfg["text_mode"], {})
    for framing in FRAMINGS:
        model = (
            f"ens-{framing}"
            if f"ens-{framing}" in report["models"]
            else f"mlp-{framing}"
        )
        paths = [
            results / "models" / f"mlp-{framing}-s{i}.pt"
            for i in range(cfg["mlp_seeds"])
        ]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"missing head weights {p.name} in {p.parent}; re-run `review-stars train`"
                )
        out.members[framing] = [heads.Head.load(p) for p in paths]
        out.calibrators[framing] = calibrate.from_arrays(
            cal_arrays, f"cal/{model}/temperature/"
        )
        out.qhat[framing] = float(
            conf_payload["models"][model]["temperature"]["aps"]["qhat"]
        )
        out.models[framing] = model
    return out


def review_draw(text: str) -> float:
    """A uniform draw in [0, 1) that depends only on the text."""
    digest = hashlib.blake2b(
        text.encode("utf-8", "surrogatepass"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


@functools.cache
def get_encoder():
    """The fastest embedding backend that agrees with torch (slow to build: it self-checks)."""
    return embed.default_choice().encoder


def predict_review(
    bundle: Bundle, encoder, text: str, title: str = "", threshold: float = 0.8
) -> dict:
    built = text_mod.build_text(title, text, bundle.text_mode)
    if not built.strip():
        raise ValueError(
            "the review is empty after cleaning (an auto-generated 'Five Stars' title does not count): "
            "give it some text"
        )
    X, n_tok = encoder.encode([built])
    u = np.array([review_draw(built)])
    framings = {}
    for framing in FRAMINGS:
        preds = Predictions([h.predict(X) for h in bundle.members[framing]])
        cal = bundle.calibrators[framing]
        P = cal.proba(preds)
        aps = conformal.ApsConformal(bundle.alpha)
        aps.qhat = bundle.qhat[framing]
        mask = aps.sets(P, u)[0]
        conf = float(P[0].max())
        framings[framing] = {
            "probs": P[0].tolist(),
            "raw_probs": preds.proba()[0].tolist(),
            "star": int(P[0].argmax()) + 1,
            "expected": float(expected_star(P)[0]),
            "confidence": conf,
            "set": [int(k) + 1 for k in np.flatnonzero(mask)],
            "abstain": conf < threshold,
            "T": cal.T,
            "model": bundle.models[framing],
        }
    return {
        "text": built,
        "n_tokens": int(n_tok[0]),
        "truncated": int(n_tok[0]) >= embed.MAX_LEN,
        "alpha": bundle.alpha,
        "threshold": threshold,
        "framings": framings,
    }
