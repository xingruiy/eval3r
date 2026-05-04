"""eval3r metric ... — compute geometry metrics from a prediction directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import typer

from eval3r.align import AlignMode
from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.metrics.geometry import ChamferVariant, evaluate_geometry
from eval3r.metrics.sampling import SampleMethod
from eval3r.prediction.reader import PredictionReader
from eval3r.report.table import print_geometry_result
from eval3r.utils.errors import MissingArtifactError

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_geom(path: str):  # type: ignore[no-untyped-def]
    p = Path(path)
    if p.is_dir():
        reader = PredictionReader(p)
        try:
            return reader.mesh
        except MissingArtifactError:
            return reader.points
    suffix = p.suffix.lower()
    if suffix not in {".ply", ".obj", ".stl", ".off", ".glb"}:
        raise typer.BadParameter(f"Unsupported geometry file extension: {suffix}")
    try:
        return load_mesh(p)
    except Exception:
        return load_point_cloud(p)


@app.command("all")
def all_cmd(
    pred: str = typer.Argument(..., help="Prediction directory or geometry file."),
    gt: str = typer.Option(..., "--gt", help="Ground-truth geometry file (ply/obj/...)."),
    samples: int = typer.Option(200_000, help="Number of samples for metric evaluation."),
    seed: int = typer.Option(42, help="RNG seed for sampling."),
    sample_method: str = typer.Option("area", help="area | vertex | uniform"),
    align: str = typer.Option("none", help="Alignment mode: none | scale | se3 | sim3 | icp"),
    thresholds: list[float] = typer.Option([0.05], "--thresholds", help="F-score thresholds."),
    chamfer_variant: str = typer.Option(
        "l1_mean_bidirectional",
        help="Chamfer variant: " + " | ".join(get_args(ChamferVariant)),
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Compute chamfer, accuracy, completeness, and F-score for a prediction vs. GT."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    if align not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if sample_method not in get_args(SampleMethod):
        raise typer.BadParameter(f"--sample-method must be one of {get_args(SampleMethod)}")
    if chamfer_variant not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")

    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        sample_method=sample_method,  # type: ignore[arg-type]
        align_mode=align,  # type: ignore[arg-type]
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,  # type: ignore[arg-type]
    )
    print_geometry_result(result, as_json=json_out)


@app.command("chamfer")
def chamfer_cmd(
    pred: str = typer.Argument(...),
    gt: str = typer.Option(..., "--gt"),
    samples: int = typer.Option(200_000),
    seed: int = typer.Option(42),
    align: str = typer.Option("none"),
    chamfer_variant: str = typer.Option("l1_mean_bidirectional"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Just chamfer distance, with thresholds=[]."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        align_mode=align,  # type: ignore[arg-type]
        thresholds=[],
        chamfer_variant=chamfer_variant,  # type: ignore[arg-type]
    )
    if json_out:
        print(json.dumps({"chamfer": result.chamfer, "variant": result.chamfer_variant}, indent=2))
    else:
        print_geometry_result(result, as_json=False)


@app.command("fscore")
def fscore_cmd(
    pred: str = typer.Argument(...),
    gt: str = typer.Option(..., "--gt"),
    threshold: float = typer.Option(0.05, help="F-score distance threshold."),
    samples: int = typer.Option(200_000),
    seed: int = typer.Option(42),
    align: str = typer.Option("none"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """F-score / precision / recall at a single threshold."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        align_mode=align,  # type: ignore[arg-type]
        thresholds=[threshold],
    )
    print_geometry_result(result, as_json=json_out)
