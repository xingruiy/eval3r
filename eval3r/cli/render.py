"""eval3r render ... — mesh / pointcloud / compare / error renderings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from eval3r.io.geometry import load_mesh, load_point_cloud

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _load_pose(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"--pose file not found: {p}")
    if p.suffix == ".npy":
        return np.load(p).reshape(4, 4).astype(np.float64)
    if p.suffix == ".json":
        return np.asarray(json.loads(p.read_text()), dtype=np.float64).reshape(4, 4)
    raise typer.BadParameter(f"--pose must be a .npy or .json file, got {p.suffix}")


@app.command("mesh")
def mesh_cmd(
    path: str = typer.Argument(..., help="Mesh file (ply/obj/...)."),
    out: str = typer.Option("render.png", "--out", help="Output image path."),
    pose: str | None = typer.Option(None, "--pose", help="4x4 OpenGL camera-to-world pose (.npy/.json)."),
    width: int = typer.Option(640),
    height: int = typer.Option(480),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
) -> None:
    from eval3r.render.pyrender_backend import render_geometry

    geom = load_mesh(path)
    render_geometry(
        geom,
        out_path=out,
        image_size=(width, height),
        camera_pose=_load_pose(pose),
        headless=headless,
    )
    typer.echo(f"wrote {out}")


@app.command("pointcloud")
def pointcloud_cmd(
    path: str = typer.Argument(..., help="Point-cloud file (ply)."),
    out: str = typer.Option("render.png", "--out"),
    pose: str | None = typer.Option(None, "--pose"),
    width: int = typer.Option(640),
    height: int = typer.Option(480),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
) -> None:
    from eval3r.render.pyrender_backend import render_geometry

    geom = load_point_cloud(path)
    render_geometry(
        geom,
        out_path=out,
        image_size=(width, height),
        camera_pose=_load_pose(pose),
        headless=headless,
    )
    typer.echo(f"wrote {out}")


@app.command("compare")
def compare_cmd(
    pred: str = typer.Argument(..., help="Prediction geometry file."),
    gt: str = typer.Argument(..., help="Ground-truth geometry file."),
    out: str = typer.Option("compare.png", "--out"),
    pose: str | None = typer.Option(None, "--pose"),
    width: int = typer.Option(640),
    height: int = typer.Option(480),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
) -> None:
    from eval3r.render.pyrender_backend import render_compare

    pred_geom = _load_any(pred)
    gt_geom = _load_any(gt)
    render_compare(
        pred_geom,
        gt_geom,
        out_path=out,
        image_size=(width, height),
        camera_pose=_load_pose(pose),
        headless=headless,
    )
    typer.echo(f"wrote {out}")


@app.command("error")
def error_cmd(
    pred: str = typer.Argument(..., help="Prediction geometry file."),
    gt: str = typer.Argument(..., help="Ground-truth geometry file."),
    out: str = typer.Option("error.ply", "--out", help="Output vertex-coloured PLY."),
    threshold: float = typer.Option(0.05, help="Distance threshold for the colour ramp."),
) -> None:
    from eval3r.render.error_map import render_error_ply

    pred_geom = _load_any(pred)
    gt_geom = _load_any(gt)
    render_error_ply(pred_geom, gt_geom, out_path=out, threshold=threshold)
    typer.echo(f"wrote {out}")


def _load_any(path: str):  # type: ignore[no-untyped-def]
    try:
        return load_mesh(path)
    except Exception:
        return load_point_cloud(path)
