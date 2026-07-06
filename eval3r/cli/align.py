"""`e3r align` command: standalone SE3/Sim3 alignment with mandatory visualization.

Estimates the transform mapping a prediction onto ground truth when the two live in
different coordinate frames/scales, using exactly two solvers (no feature-based
global registration): closest-point ICP on the geometries, or trajectory-first
Umeyama propagation. Always writes the before/after overlay PLYs, the orthographic
projection PNG, ``pred_aligned.ply``, and ``alignment.json`` — a residual number
alone does not show a flipped, mirrored, or locally-stuck registration.

Deliberately verbose (``rich``): echoes the resolved configuration (including an
``auto``-resolved max correspondence distance) and prints full failure reasons,
never a bare exit code (CLAUDE.md CLI verbosity rules).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.core.errors import Eval3rError

console = Console()
err_console = Console(stderr=True)

_MODE_CHOICES = ("se3", "sim3")
_SOLVER_CHOICES = ("icp", "trajectory")
_KIND_CHOICES = ("pointcloud", "mesh")


def _default_out_dir(pred: Path) -> Path:
    # UTC for the default directory name so the local timezone is not leaked.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return Path("runs") / f"{stamp}_align_{pred.stem or 'geometry'}"


def align_command(
    pred: Path = typer.Argument(..., help="Prediction geometry to align (mesh or point cloud)."),
    gt: Path = typer.Option(..., "--gt", help="Ground-truth geometry to align onto."),
    mode: str = typer.Option(
        "se3", "--mode", help="Transform class: se3 (rigid) | sim3 (with scale)."
    ),
    solver: str = typer.Option(
        "icp", "--solver",
        help="icp (closest-point ICP on the geometries) | trajectory (align the "
        "predicted camera trajectory onto the GT trajectory, then propagate).",
    ),
    max_corr_dist: str | None = typer.Option(
        None, "--max-corr-dist",
        help="ICP max correspondence distance in metres, or 'auto' (5% of the GT "
        "bounding-box diagonal; the resolved value is echoed and recorded). "
        "Required for --solver icp.",
    ),
    max_iterations: int = typer.Option(
        50, "--max-iterations", help="ICP iteration cap (recorded)."
    ),
    max_points: int = typer.Option(
        200_000, "--max-points",
        help="Per-side subsample cap for ICP estimation (seeded, recorded).",
    ),
    pred_trajectory: Path | None = typer.Option(
        None, "--pred-trajectory",
        help="Predicted camera trajectory (TUM format; required for --solver trajectory).",
    ),
    gt_trajectory: Path | None = typer.Option(
        None, "--gt-trajectory",
        help="Ground-truth camera trajectory (TUM format; required for --solver trajectory).",
    ),
    associate_max_diff: float | None = typer.Option(
        None, "--associate-max-diff",
        help="Timestamp association tolerance in seconds (required for --solver "
        "trajectory; never defaulted).",
    ),
    input_type: str = typer.Option(
        "pointcloud", "--input", help="Prediction type: pointcloud | mesh."
    ),
    gt_type: str = typer.Option(
        "pointcloud", "--gt-input", help="Ground-truth type: pointcloud | mesh."
    ),
    out: Path | None = typer.Option(
        None, "--out",
        help="Output directory (default: runs/<timestamp>_align_<pred stem>).",
    ),
) -> None:
    """Align a prediction onto ground truth (SE3/Sim3) and visualize the result."""
    for label, value, choices in (
        ("--mode", mode, _MODE_CHOICES),
        ("--solver", solver, _SOLVER_CHOICES),
        ("--input", input_type, _KIND_CHOICES),
        ("--gt-input", gt_type, _KIND_CHOICES),
    ):
        if value not in choices:
            err_console.print(
                f"[bold red]invalid {label} '{value}'[/]: choose one of {', '.join(choices)}."
            )
            raise typer.Exit(code=2)

    mcd: float | str | None = None
    if max_corr_dist is not None:
        if max_corr_dist == "auto":
            mcd = "auto"
        else:
            try:
                mcd = float(max_corr_dist)
            except ValueError:
                err_console.print(
                    f"[bold red]invalid --max-corr-dist '{max_corr_dist}'[/]: pass a "
                    f"distance in metres or 'auto'."
                )
                raise typer.Exit(code=2) from None

    out_dir = Path(out) if out is not None else _default_out_dir(Path(pred))

    from eval3r.api import align_geometries

    try:
        output = align_geometries(
            pred, gt,
            out_dir=out_dir,
            mode=mode, solver=solver,
            input_type=input_type,  # type: ignore[arg-type]
            gt_type=gt_type,  # type: ignore[arg-type]
            max_corr_dist=mcd,
            max_iterations=max_iterations, max_points=max_points,
            pred_trajectory=pred_trajectory, gt_trajectory=gt_trajectory,
            associate_max_diff=associate_max_diff,
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="alignment failed", style="red", expand=False))
        err_console.print_exception()
        raise typer.Exit(code=1) from exc

    alignment = output.alignment
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("prediction", f"{pred}  [{input_type}]")
    table.add_row("ground truth", f"{gt}  [{gt_type}]")
    table.add_row("mode / solver", f"{alignment['mode']} / {alignment['solver']}")
    table.add_row("estimate on", str(alignment["estimate_on"]))
    params = alignment.get("parameters", {})
    if solver == "icp":
        table.add_row(
            "max corr dist",
            f"{params.get('max_correspondence_distance'):.6g} m"
            + ("  (auto: " + params["max_corr_dist_auto_rule"] + ")"
               if params.get("max_corr_dist_auto") else ""),
        )
        table.add_row("fitness", f"{alignment.get('fitness'):.4f}")
    else:
        table.add_row("pred trajectory", str(pred_trajectory))
        table.add_row("gt trajectory", str(gt_trajectory))
        table.add_row(
            "association",
            f"max_diff={params.get('associate_max_diff')}s, "
            f"associated={alignment.get('n_correspondences')}",
        )
    table.add_row("scale", f"{alignment['scale']:.6g}")
    residual = alignment.get("residual_rmse")
    table.add_row("residual RMSE", f"{residual:.6g} m" if residual is not None else "-")
    table.add_row("output dir", str(output.out_dir))
    console.print(Panel(table, title="alignment estimated", expand=False))

    console.print(f"[green]aligned prediction written:[/] {output.pred_aligned}")
    console.print(f"[green]transform + provenance:[/] {output.alignment_json}")
    console.print(
        f"[green]visualization:[/] {output.out_dir / output.vis_manifest['before_ply']}, "
        f"{output.out_dir / output.vis_manifest['after_ply']}, "
        f"{output.out_dir / output.vis_manifest['projections_png']}"
    )
    console.print(
        "[yellow]inspect the overlays[/] (pred = red, gt = blue): a small residual can "
        "still hide a flipped or locally-stuck registration."
    )
