"""Command-line interface: fetch -> embed -> evaluate -> dedupe.

uv run duplicate-listings fetch --groups 1400
uv run duplicate-listings embed
uv run duplicate-listings evaluate
uv run duplicate-listings dedupe --split test --out duplicates.csv
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from duplicate_listings import data, evaluate, pipeline

app = typer.Typer(
    add_completion=False,
    help="Find duplicate product listings with text + image embeddings.",
)

DataDir = Annotated[
    Path,
    typer.Option(
        "--data-dir", help="Where the catalog, images, embeddings and index live."
    ),
]


class SplitName(str, Enum):
    val = "val"
    test = "test"


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


@app.command()
def fetch(
    groups: Annotated[
        int, typer.Option(min=1, help="How many whole duplicate groups to keep.")
    ] = 1400,
    seed: Annotated[int, typer.Option(help="Seed for choosing which groups.")] = 0,
    data_dir: DataDir = data.DATA_DIR,
) -> None:
    """Download the Shopee catalog and the images for a group-aware slice of it."""
    df = pipeline.build_slice(groups, seed, data_dir)
    typer.echo(
        f"slice: {df.height} listings in {df['label_group'].n_unique()} duplicate groups -> {data_dir / pipeline.SLICE_FILE}"
    )


@app.command()
def embed(data_dir: DataDir = data.DATA_DIR) -> None:
    """Embed every listing's title (multilingual-e5) and image (SigLIP 2); cached."""
    try:
        df = pipeline.load_slice(data_dir)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    emb = pipeline.embed_slice(df, data_dir)
    typer.echo(
        f"embedded {len(emb.posting_ids)} listings: text {emb.text.shape[1]}d, image {emb.image.shape[1]}d"
    )
    if emb.missing_images:
        typer.echo(
            f"warning: {len(emb.missing_images)} listings have no readable image"
        )


@app.command("evaluate")
def evaluate_cmd(
    k: Annotated[
        int, typer.Option(min=1, help="Neighbours retrieved per listing per modality.")
    ] = 10,
    val_fraction: Annotated[float, typer.Option(min=0.05, max=0.95)] = 0.4,
    seed: Annotated[
        int, typer.Option(help="Seed for the validation/test group split.")
    ] = 0,
    data_dir: DataDir = data.DATA_DIR,
) -> None:
    """Tune on validation groups, report text / image / fused scoring on test groups."""
    try:
        df = pipeline.load_slice(data_dir)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    emb = pipeline.embed_slice(df, data_dir)
    report, _, _ = pipeline.run_evaluation(df, emb, data_dir, k, val_fraction, seed)
    pipeline.save_report(
        data_dir / pipeline.REPORT_FILE,
        report,
        settings={"k": k, "val_fraction": val_fraction, "seed": seed},
    )
    typer.echo(
        f"test split: {report.n_listings} listings, {report.n_true_pairs} true duplicate pairs"
    )
    typer.echo(
        f"candidate recall: {report.candidate_recall:.3f} ({report.n_candidates} candidate pairs)"
    )
    typer.echo("")
    typer.echo(report.to_markdown())


@app.command()
def dedupe(
    split: Annotated[
        SplitName, typer.Option(help="Which held-out split to cluster.")
    ] = SplitName.test,
    out: Annotated[
        Path, typer.Option(help="CSV to write the duplicate clusters to.")
    ] = Path("duplicates.csv"),
    data_dir: DataDir = data.DATA_DIR,
) -> None:
    """Write predicted duplicate clusters using the fused weight/threshold tuned by `evaluate`."""
    try:
        report = pipeline.load_report(data_dir / pipeline.REPORT_FILE)
        df = pipeline.load_slice(data_dir)
    except FileNotFoundError as exc:
        raise _fail(exc) from exc
    emb = pipeline.embed_slice(df, data_dir)
    s = report["settings"]
    val, test = pipeline.prepare_splits(
        df, emb, data_dir, s["k"], s["val_fraction"], s["seed"]
    )
    chosen = val if split is SplitName.val else test
    fused = report["modes"]["fused"]
    table = evaluate.duplicate_clusters(chosen, fused["weight"], fused["threshold"])
    table.write_csv(out)
    typer.echo(
        f"{table['cluster_id'].n_unique()} duplicate clusters ({table.height} listings) -> {out}"
    )


if __name__ == "__main__":
    app()
