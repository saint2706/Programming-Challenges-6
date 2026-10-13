"""``scorecard`` command line: train, score one applicant, list/benchmark OpenVINO devices."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from credit_scorecard import data, ov_scoring, pipeline, report
from credit_scorecard.paths import project_root
from credit_scorecard.scorecard import Scorecard

DEFAULT_OUT = "results"


def _results_dir(value: str) -> Path:
    return Path(value) if Path(value).is_absolute() else project_root() / value


def cmd_train(args: argparse.Namespace) -> int:
    result = pipeline.run(
        args.seed,
        max_bins=args.max_bins,
        min_bin_frac=args.min_bin_frac,
        base_score=args.base_score,
        base_odds=args.base_odds,
        pdo=args.pdo,
    )
    path = report.write_report(result, _results_dir(args.out))
    m = result.metrics
    print(
        f"scorecard   test AUC {m['scorecard_test']['auc']:.4f}  Gini {m['scorecard_test']['gini']:.4f}  KS {m['scorecard_test']['ks']:.4f}"
    )
    print(
        f"gradient bt test AUC {m['gbm_test']['auc']:.4f}  Gini {m['gbm_test']['gini']:.4f}  KS {m['gbm_test']['ks']:.4f}"
    )
    print(
        f"score PSI train vs test {m['score_psi_train_vs_test']:.4f}; features: {', '.join(m['features_used'])}"
    )
    print(f"report: {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    card = Scorecard.load(_results_dir(args.model))
    text = (
        Path(args.applicant_file).read_text(encoding="utf-8")
        if args.applicant_file
        else args.applicant
    )
    try:
        applicant = json.loads(text)
        if not isinstance(applicant, dict):
            raise TypeError("applicant must be a JSON object")
        unknown = set(applicant) - set(data.FEATURES)
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
    except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    frame = pl.DataFrame(
        {f: [applicant.get(f)] for f in card.features},
        schema={f: pl.Float64 for f in card.features},
    )
    pts = {f: int(p[0]) for f, p in card.feature_points(frame).items()}
    score = sum(pts.values())
    print(
        json.dumps(
            {
                "score": score,
                "probability_of_default": round(float(card.probability_bad(score)), 5),
                "points": pts,
                "unspecified_fields": [
                    f for f in card.features if applicant.get(f) is None
                ],
                "reason_codes": [
                    {"feature": f, "points_lost": lost}
                    for f, lost in card.reason_codes(applicant)
                ],
            },
            indent=2,
        )
    )
    return 0


def cmd_devices(_: argparse.Namespace) -> int:
    for name, full in ov_scoring.available_devices().items():
        print(f"{name:5} {full}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    card, _, test = pipeline.train_scorecard()
    frame = pl.concat([test] * max(1, args.rows // test.height))
    exact = card.score(frame)

    def timed(fn) -> float:
        fn()
        start = time.perf_counter()
        for _ in range(3):
            fn()
        return (time.perf_counter() - start) / 3 * 1000

    print(f"{frame.height:,} rows")
    print(f"{'numpy':8} {timed(lambda: card.score(frame)):8.1f} ms")
    for device in args.device:
        scorer = ov_scoring.OvScorer(card, device)
        x = scorer.matrix(frame)
        parity = float(np.mean(scorer.score(x) == exact))
        print(
            f"{device:8} {timed(lambda s=scorer, x=x: s.score(x)):8.1f} ms   parity with numpy {parity:.4%}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scorecard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit the scorecard and write the report")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--out", default=DEFAULT_OUT, help="output folder (default: results/)"
    )
    train.add_argument("--max-bins", type=int, default=8)
    train.add_argument("--min-bin-frac", type=float, default=0.02)
    train.add_argument("--base-score", type=float, default=600)
    train.add_argument(
        "--base-odds", type=float, default=50, help="good:bad odds at the base score"
    )
    train.add_argument(
        "--pdo", type=float, default=20, help="points to double the odds"
    )
    train.set_defaults(func=cmd_train)

    score = sub.add_parser("score", help="score one applicant given as JSON")
    score.add_argument("--model", default=f"{DEFAULT_OUT}/scorecard.json")
    who = score.add_mutually_exclusive_group(required=True)
    who.add_argument(
        "--applicant", help='JSON object, e.g. \'{"age": 41, "MonthlyIncome": 5200}\''
    )
    who.add_argument("--applicant-file", help="path to a JSON file with one applicant")
    score.set_defaults(func=cmd_score)

    sub.add_parser("devices", help="list OpenVINO devices").set_defaults(
        func=cmd_devices
    )

    bench = sub.add_parser(
        "bench", help="time batch scoring: numpy vs OpenVINO devices"
    )
    bench.add_argument("--rows", type=int, default=4_000_000)
    bench.add_argument("--device", nargs="+", default=["CPU", "GPU"])
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
