"""Command line: pivot a CSV, cross-tabulate two fields, drill into a cell, export an interactive page."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table as RichTable

from pivot_tool import render
from pivot_tool.engine import Table, read_bytes
from pivot_tool.ffi import BuildError, EngineError, build_library
from pivot_tool.html_export import WASM_NAME, build_html
from pivot_tool.paths import project_root
from pivot_tool.presets import DATASETS
from pivot_tool.spec import SpecError, build_spec, parse_dim, parse_filter

app = typer.Typer(
    help="A pivot table and cross-tabulation engine (Rust) with a command line and an HTML export.",
    no_args_is_help=True,
)
console = Console()

FileArg = Annotated[
    Path,
    typer.Argument(help="A CSV file (.csv or .csv.gz).", exists=True, dir_okay=False),
]
RowsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--rows",
        "-r",
        help="A row field: name, name:month, name:bin=10 ... (repeatable).",
    ),
]
ColsOpt = Annotated[
    list[str] | None, typer.Option("--cols", "-c", help="A column field (repeatable).")
]
ValuesOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--values",
        "-v",
        help="agg(field) [as label] [@show_as] or calc: formula (repeatable).",
    ),
]
FiltersOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--filter",
        "-f",
        help="field=value, age>=30, x in a|b, x between 1..5 ... (repeatable).",
    ),
]
DelimOpt = Annotated[
    str | None,
    typer.Option(help="Field delimiter; guessed from the first line by default."),
]


def fail(message: str, code: int = 2) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code)


def load(path: Path, delimiter: str | None = None) -> Table:
    try:
        return Table(read_bytes(path), delimiter=delimiter)
    except (EngineError, BuildError) as exc:
        raise fail(str(exc)) from exc


@app.command()
def describe(file: FileArg, delimiter: DelimOpt = None) -> None:
    """The fields of a file: type, missing values, distinct values, range."""
    with load(file, delimiter) as t:
        table = RichTable(title=f"{file.name}: {t.rows:,} rows")
        for c in ("field", "type", "missing", "distinct", "min", "max"):
            table.add_column(c, justify="left" if c in ("field", "type") else "right")
        for f in t.fields:
            table.add_row(
                f["name"],
                f["kind"],
                f"{f['nulls']:,}",
                "" if f["distinct"] is None else f"{f['distinct']:,}",
                f["min"] or "",
                f["max"] or "",
            )
        console.print(table)


@app.command()
def pivot(
    file: FileArg,
    rows: RowsOpt = None,
    cols: ColsOpt = None,
    values: ValuesOpt = None,
    filters: FiltersOpt = None,
    subtotals: Annotated[
        bool,
        typer.Option(help="A subtotal for every group of an axis with several fields."),
    ] = False,
    totals_row: Annotated[bool, typer.Option(help="A grand-total row.")] = True,
    totals_col: Annotated[bool, typer.Option(help="A grand-total column.")] = True,
    sort: Annotated[
        str | None,
        typer.Option(help="First row field: label[:desc] or value:MEASURE[:asc|desc]."),
    ] = None,
    top: Annotated[
        str | None, typer.Option(help="First row field: keep the top N:MEASURE.")
    ] = None,
    spec_file: Annotated[
        Path | None,
        typer.Option(
            "--spec",
            help="A full JSON spec instead of the options above.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="Write the result to .csv, .md, .json or .html instead of printing it."
        ),
    ] = None,
    delimiter: DelimOpt = None,
) -> None:
    """Pivot a file: any number of row and column fields, ten-odd aggregations, subtotals, sorting, top-N."""
    try:
        spec = (
            json.loads(spec_file.read_text(encoding="utf-8"))
            if spec_file
            else build_spec(
                rows,
                cols,
                values,
                filters,
                subtotals=subtotals,
                totals_row=totals_row,
                totals_col=totals_col,
                sort=sort,
                top=top,
            )
        )
    except (SpecError, json.JSONDecodeError) as exc:
        raise fail(str(exc)) from exc
    with load(file, delimiter) as t:
        try:
            result = t.pivot(spec)
        except EngineError as exc:
            raise fail(str(exc)) from exc
        if out is not None:
            suffix = out.suffix.lower()
            if suffix == ".html":
                page = build_html(read_bytes(file), file.name, initial=spec)
                out.write_text(page, encoding="utf-8", newline="\n")
            elif suffix == ".json":
                out.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
                )
            elif suffix == ".md":
                out.write_text(
                    render.to_markdown(result), encoding="utf-8", newline="\n"
                )
            else:
                out.write_text(render.to_csv(result), encoding="utf-8", newline="\n")
            console.print(
                f"{len(result['rows'])} rows x {len(result['cols'])} columns -> {out}"
            )
            return
    console.print(
        render.to_rich(
            result, title=f"{result['rows_used']:,} of {result['rows_total']:,} rows"
        )
    )
    for w in result["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {w}")


@app.command()
def crosstab(
    file: FileArg,
    rows: Annotated[
        str,
        typer.Option(
            "--rows", "-r", help="The row field (may be grouped: age:bin=10)."
        ),
    ],
    cols: Annotated[str, typer.Option("--cols", "-c", help="The column field.")],
    layer: Annotated[
        str | None,
        typer.Option(
            help="A third field: repeat the table inside each of its values (Simpson's paradox check for 2x2)."
        ),
    ] = None,
    filters: FiltersOpt = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the raw result as JSON.")
    ] = False,
    delimiter: DelimOpt = None,
) -> None:
    """Counts of two categorical fields and the statistics of their association."""
    try:
        spec: dict = {
            "row": parse_dim(rows),
            "col": parse_dim(cols),
            "filters": [parse_filter(f) for f in filters or []],
        }
        if layer:
            spec["layer"] = parse_dim(layer)
    except SpecError as exc:
        raise fail(str(exc)) from exc
    with load(file, delimiter) as t:
        try:
            result = t.crosstab(spec)
        except EngineError as exc:
            raise fail(str(exc)) from exc
    if as_json:
        console.print_json(json.dumps(result))
        return
    for item in render.crosstab_tables(result):
        console.print(item)


@app.command()
def drill(
    file: FileArg,
    rows: RowsOpt = None,
    cols: ColsOpt = None,
    filters: FiltersOpt = None,
    row_path: Annotated[
        list[str] | None,
        typer.Option(
            help="The row labels that name the cell, outermost first (repeatable)."
        ),
    ] = None,
    col_path: Annotated[
        list[str] | None,
        typer.Option(help="The column labels that name the cell (repeatable)."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Show at most this many rows.")] = 20,
    delimiter: DelimOpt = None,
) -> None:
    """The source rows behind one cell of a pivot."""
    try:
        spec = build_spec(rows, cols, [], filters)
    except SpecError as exc:
        raise fail(str(exc)) from exc
    with load(file, delimiter) as t:
        try:
            result = t.drill(spec, row_path, col_path, limit)
        except EngineError as exc:
            raise fail(str(exc)) from exc
    table = RichTable(
        title=f"{result['total']:,} rows behind the cell (showing {len(result['rows'])})"
    )
    for f in result["fields"]:
        table.add_column(f, overflow="fold")
    for r in result["rows"]:
        table.add_row(*[c if c is not None else "" for c in r])
    console.print(table)


@app.command()
def html(
    file: FileArg,
    out: Annotated[Path, typer.Option(help="The page to write.")] = Path("pivot.html"),
    title: Annotated[
        str | None, typer.Option(help="Page title (default: the file name).")
    ] = None,
    rows: RowsOpt = None,
    cols: ColsOpt = None,
    values: ValuesOpt = None,
    filters: FiltersOpt = None,
) -> None:
    """Write one self-contained page that pivots this file in the browser (the engine runs there as WebAssembly)."""
    try:
        initial = build_spec(rows, cols, values, filters)
    except SpecError as exc:
        raise fail(str(exc)) from exc
    with load(file) as t:
        names = ", ".join(t.names[:6]) + (", ..." if len(t.names) > 6 else "")
        page = build_html(read_bytes(file), title or file.name, initial=initial)
    out.write_text(page, encoding="utf-8", newline="\n")
    console.print(f"{len(page) / 1e6:.2f} MB -> {out}  (fields: {names})")


@app.command()
def demo(
    dataset: Annotated[
        str, typer.Argument(help="adult (UCI census) or berkeley (1973 admissions).")
    ] = "adult",
    html_out: Annotated[
        Path | None,
        typer.Option(
            "--html", help="Also write the interactive page with these presets."
        ),
    ] = None,
) -> None:
    """Run the bundled pivots on a bundled dataset."""
    ds = DATASETS.get(dataset)
    if ds is None:
        raise fail(f"unknown dataset '{dataset}': choose {', '.join(DATASETS)}")
    with load(ds.path) as t:
        for preset in ds.presets:
            console.rule(preset["name"])
            console.print(f"[dim]{preset['description']}[/dim]")
            spec = preset["spec"]
            try:
                if preset.get("tab") == "crosstab":
                    ct_spec = {"row": spec["rows"][0], "col": spec["cols"][0]}
                    if preset.get("layer"):
                        ct_spec["layer"] = {"field": preset["layer"]}
                    for item in render.crosstab_tables(t.crosstab(ct_spec)):
                        console.print(item)
                else:
                    console.print(render.to_rich(t.pivot(spec)))
            except EngineError as exc:
                raise fail(str(exc)) from exc
    if html_out is not None:
        page = build_html(read_bytes(ds.path), ds.title, presets=ds.presets)
        html_out.write_text(page, encoding="utf-8", newline="\n")
        console.print(f"{len(page) / 1e6:.2f} MB -> {html_out}")


@app.command()
def build(
    wasm: Annotated[
        bool,
        typer.Option(
            help="Also build the WebAssembly module and copy it into the package."
        ),
    ] = False,
) -> None:
    """Build the Rust engine with cargo (needed once; the CLI does it automatically if missing)."""
    try:
        path = build_library()
        console.print(f"engine -> {path}")
        if wasm:
            built = build_library(wasm=True)
            target = project_root() / "src" / "pivot_tool" / "web" / WASM_NAME
            shutil.copyfile(built, target)
            console.print(f"wasm -> {target} ({target.stat().st_size / 1024:.0f} KB)")
    except BuildError as exc:
        raise fail(str(exc)) from exc


@app.command()
def bench(
    rows: Annotated[
        int,
        typer.Option(
            help="Rows of the synthetic table (the Adult census, repeated and perturbed)."
        ),
    ] = 2_000_000,
    repeat: Annotated[
        int, typer.Option(help="Timings are the best of this many runs.")
    ] = 3,
) -> None:
    """Time the engine against polars and DuckDB on the same pivots (needs the dev dependencies)."""
    from pivot_tool.bench import run_benchmark

    run_benchmark(console, rows, repeat)
