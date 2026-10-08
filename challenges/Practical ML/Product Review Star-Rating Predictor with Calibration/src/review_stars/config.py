"""Run configuration, and the paths everything else derives from.

Every path hangs off ``project_root()`` (the challenge folder, or ``REVIEW_STARS_HOME`` when set),
never off the current directory, so the CLI, Streamlit and the tests all agree on where ``data/``
and ``results/`` live.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_MODES = ("title_text", "text_only", "raw_title")
FRAMINGS = ("classification", "regression", "ordinal")


def project_root() -> Path:
    return Path(
        os.environ.get("REVIEW_STARS_HOME") or Path(__file__).resolve().parents[2]
    )


def data_dir() -> Path:
    return project_root() / "data"


def results_dir() -> Path:
    return project_root() / "results"


@dataclass(frozen=True)
class Config:
    # data: two categories, time windows counted back from the common end of the data
    in_domain: str = "Appliances"
    out_domain: str = "Software"
    n_train: int = 150_000
    n_val: int = 20_000
    n_cal: int = 20_000
    n_test: int = 30_000
    n_ood: int = 30_000
    n_ood_pool: int = 5_000
    test_months: int = 12
    cal_months: int = 6
    val_months: int = 6
    min_month_rows: int = 100
    data_seed: int = 0
    text_mode: str = "title_text"
    # heads
    l2_grid: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    mlp_hidden: int = 512
    mlp_dropouts: tuple[float, ...] = (0.1, 0.3)
    mlp_seeds: int = 5
    mlp_epochs: int = 40
    mlp_patience: int = 4
    mlp_lr: float = 1e-3
    mlp_batch: int = 512
    mlp_weight_decay: float = 1e-4
    tfidf_features: int = 200_000
    tfidf_c_grid: tuple[float, ...] = (1.0, 4.0, 16.0)
    ridge_alpha_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
    head_seed: int = 0
    # calibration study
    n_boot: int = 1000
    ece_bins: int = 15
    alpha: float = 0.1
    levels: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    recal_ns: tuple[int, ...] = (50, 100, 250, 500, 1000, 2500, 5000)
    recal_draws: int = 20
    ablation_train: int = 40_000
    stat_seed: int = 0
    # how the work is run, never what it computes: excluded from every result key
    n_jobs: int = 1

    def __post_init__(self):
        if self.text_mode not in TEXT_MODES:
            raise ValueError(
                f"text_mode must be one of {TEXT_MODES}, not {self.text_mode!r}"
            )
        positive = [
            "n_train",
            "n_val",
            "n_cal",
            "n_test",
            "n_ood",
            "n_ood_pool",
            "test_months",
            "cal_months",
            "val_months",
            "mlp_seeds",
            "mlp_epochs",
            "mlp_batch",
            "mlp_hidden",
            "n_boot",
            "recal_draws",
            "ablation_train",
            "n_jobs",
        ]
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, not {getattr(self, name)}"
                )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"alpha must be strictly between 0 and 1, not {self.alpha}"
            )
        if self.ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")

    def to_dict(self) -> dict:
        """JSON-ready (tuples become lists) and without run-only settings."""
        out = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()
        }
        out.pop("n_jobs")
        return out
