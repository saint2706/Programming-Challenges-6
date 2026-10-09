"""Hand-built DuckDB profiler JSON, so plan and rule tests need no database."""

from __future__ import annotations


def op(
    name: str,
    rows: int = 0,
    est: int | str | None = None,
    time: float = 0.0,
    scanned: int = 0,
    children: list | None = None,
    **info,
) -> dict:
    """One operator in DuckDB's profile format (``extra_info`` keys are passed as keywords)."""
    extra = {k.replace("_", " ").title(): v for k, v in info.items()}
    if est is not None:
        extra["Estimated Cardinality"] = str(est)
    return {
        "operator_type": name,
        "operator_name": "SEQ_SCAN " if name == "TABLE_SCAN" else name,
        "operator_cardinality": rows,
        "operator_timing": time,
        "operator_rows_scanned": scanned,
        "result_set_size": rows * 8,
        "extra_info": extra,
        "children": children or [],
    }


def scan(
    table: str,
    rows: int,
    scanned: int | None = None,
    est: int | None = None,
    time: float = 0.0,
    **info,
) -> dict:
    return op(
        "TABLE_SCAN",
        rows=rows,
        est=est if est is not None else rows,
        time=time,
        scanned=rows if scanned is None else scanned,
        Table=f"memory.main.{table}",
        **info,
    )


def wrap(root: dict, latency: float = 0.1, cpu: float = 0.2) -> dict:
    """The JSON of ``EXPLAIN (ANALYZE, FORMAT JSON)``: a wrapper node above the real root."""
    return {
        "latency": latency,
        "cpu_time": cpu,
        "system_peak_buffer_memory": 1_000_000,
        "children": [
            {
                "operator_type": "EXPLAIN_ANALYZE",
                "operator_name": "EXPLAIN_ANALYZE",
                "children": [root],
            }
        ],
    }
