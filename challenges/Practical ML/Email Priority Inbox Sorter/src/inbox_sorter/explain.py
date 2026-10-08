"""Per-message reasons from SHAP attributions on the LightGBM models.

``TreeExplainer`` gives exact additive contributions in log-odds space: the base
value plus every feature's contribution equals the model's raw margin. The 64 SVD
text components are merged into one "wording" reason, since an individual
component has no readable meaning.
"""

from __future__ import annotations

import warnings

import numpy as np
import shap

READABLE: dict[str, str] = {
    "n_to": "number of direct recipients",
    "n_cc": "number of people on Cc",
    "owner_in_to": "you are a direct recipient (not just Cc)",
    "mass_mail": "sent to a large group",
    "sender_is_enron": "sender is inside the company",
    "subject_len": "subject length",
    "subject_caps": "subject is in capitals",
    "body_len": "message length",
    "question_marks": "questions in the text",
    "mentions_attachment": "mentions an attachment",
    "hour": "hour it arrived",
    "weekday": "day of the week",
    "business_hours": "arrived in business hours",
    "re_depth": "depth of the reply chain",
    "fw_depth": "forwarded depth",
    "owner_in_thread": "you already wrote in this thread",
    "sender_prior_count": "how much this sender has written before",
    "sender_prior_acted": "times you answered this sender before",
    "sender_prior_acted_rate": "how often you answer this sender",
    "owner_prior_sends_to_sender": "how much you have written to this sender",
    "svd": "wording of the message",
}


def contributions(model, x: np.ndarray) -> tuple[np.ndarray, float]:
    """``(per-feature SHAP values in log-odds, base value)`` for a fitted LightGBM."""
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        # shap warns that binary LightGBM output "has changed to a list"; the
        # list/array shapes are both handled right below
        warnings.filterwarnings("ignore", message=".*output has changed to a list.*")
        values = explainer.shap_values(x)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # newer shap: (rows, features, classes)
        values = values[:, :, 1]
    base = np.atleast_1d(explainer.expected_value)
    return values, float(base[-1])


def group_contributions(
    contribs: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Readable labels per column, with every ``svd_*`` column summed into one."""
    labels = [READABLE["svd"] if n.startswith("svd_") else READABLE[n] for n in names]
    unique = list(dict.fromkeys(labels))
    out = np.zeros((contribs.shape[0], len(unique)))
    for j, label in enumerate(labels):
        out[:, unique.index(label)] += contribs[:, j]
    return out, unique


def reasons(
    model, x: np.ndarray, names: list[str], k: int = 4
) -> list[list[tuple[str, float]]]:
    """Top-``k`` ``(label, contribution)`` pairs per row, largest magnitude first."""
    contribs, _ = contributions(model, x)
    grouped, labels = group_contributions(contribs, names)
    out = []
    for row in grouped:
        top = np.argsort(-np.abs(row))[:k]
        out.append([(labels[j], float(row[j])) for j in top])
    return out
