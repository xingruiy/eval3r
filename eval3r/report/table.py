"""Render evaluation results as a rich table or JSON."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from eval3r.metrics.metric3d import EvalResult3D


def render_geometry_table(result: EvalResult3D, *, title: str | None = None) -> Table:
    table = Table(title=title or "eval3r — geometry metrics", show_lines=False)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", style="white")

    table.add_row("chamfer", f"{result.chamfer:.6f}")
    table.add_row("chamfer_variant", result.chamfer_variant)
    table.add_row("accuracy", f"{result.accuracy:.6f}")
    table.add_row("completeness", f"{result.completeness:.6f}")
    for thr in sorted(result.fscore.keys()):
        v = result.fscore[thr]
        table.add_row(f"f-score @ {thr}", f"{v['f']:.4f}  (P={v['precision']:.4f} R={v['recall']:.4f})")
    table.add_row("samples", str(result.samples))
    table.add_row("seed", str(result.seed))
    table.add_row("sample_method", result.sample_method)
    table.add_row("align_mode", result.align_mode)
    if result.align_mode != "none":
        table.add_row("align_scale", f"{result.align_scale:.6f}")
    return table


def print_geometry_result(
    result: EvalResult3D,
    *,
    as_json: bool = False,
    console: Console | None = None,
) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    (console or Console()).print(render_geometry_table(result))


from eval3r.metrics.metric2d import EvalResult2D


def render_depth_table(result: EvalResult2D, *, title: str | None = None) -> Table:
    table = Table(title=title or "eval3r — depth metrics", show_lines=False)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", style="white")

    table.add_row("abs_rel", f"{result.abs_rel:.6f}")
    table.add_row("sq_rel", f"{result.sq_rel:.6f}")
    table.add_row("rmse", f"{result.rmse:.6f}")
    table.add_row("rmse_log", f"{result.rmse_log:.6f}")
    table.add_row("delta1 (< 1.25)", f"{result.delta1:.4f}")
    table.add_row("delta2 (< 1.25²)", f"{result.delta2:.4f}")
    table.add_row("delta3 (< 1.25³)", f"{result.delta3:.4f}")
    table.add_row("valid_pixels", str(result.valid_pixels))
    table.add_row("total_pixels", str(result.total_pixels))
    return table


def print_depth_result(
    result: EvalResult2D,
    *,
    as_json: bool = False,
    console: Console | None = None,
) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    (console or Console()).print(render_depth_table(result))


def print_dict(data: dict[str, Any], *, title: str = "summary") -> None:
    table = Table(title=title)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    Console().print(table)
