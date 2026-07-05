"""Debug outputs: distance histograms and error-colored point clouds.

Written into a run directory's ``debug/`` subdirectory only when the protocol's
reporting spec explicitly requests them (``save_colored_errors`` /
``save_distance_histogram`` — both default off). The colored cloud maps each
cleaned prediction point's pred→gt nearest-neighbor distance through a colormap;
the histogram shows both distance directions. Every parameter that shaped the
outputs (colormap, normalization max, bin count, point counts) is recorded in
``debug/debug_outputs.json`` so the artifacts are interpretable later.

Only evaluation paths that produce per-point distances (the eval3r-native geometry
metric path) can emit these outputs; official-toolbox paths own their distance
computation internally and are recorded as such in the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eval3r.core.schema import ReportingSpec
from eval3r.metrics.geometry import DirectionalDistances
from eval3r.reports.json import dump_json

#: Colormap for error-colored point clouds (low error = dark blue, high = red).
ERROR_COLORMAP = "turbo"
#: Histogram bin count.
HISTOGRAM_BINS = 64


def write_distance_histogram(
    distances: DirectionalDistances,
    path: Path,
    *,
    scene_id: str,
    bins: int = HISTOGRAM_BINS,
) -> None:
    """Write a PNG histogram of pred→gt and gt→pred distances for one scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(7, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.hist(distances.pred_to_gt, bins=bins, alpha=0.6, label="pred → gt")
    ax.hist(distances.gt_to_pred, bins=bins, alpha=0.6, label="gt → pred")
    ax.set_xlabel("nearest-neighbor distance")
    ax.set_ylabel("point count")
    ax.set_title(f"distance histogram: {scene_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), format="png")


def error_colors(distances: np.ndarray, *, vmax: float) -> np.ndarray:
    """Map distances to (N, 3) uint8 colors via the error colormap, clipped at vmax."""
    if vmax <= 0.0:  # all distances zero: everything is at the colormap minimum
        normalized = np.zeros_like(distances)
    else:
        normalized = np.clip(distances / vmax, 0.0, 1.0)
    rgba = colormaps[ERROR_COLORMAP](normalized)
    return (np.asarray(rgba)[:, :3] * 255).astype(np.uint8)


def write_error_colored_pointcloud(
    distances: DirectionalDistances,
    path: Path,
    *,
    pointcloud_backend: Any,
    vmax: float,
) -> None:
    """Write the cleaned prediction points colored by pred→gt distance.

    ``vmax`` is the distance mapped to the top of the colormap (larger distances
    are clipped); the caller records it in the debug manifest.
    """
    if distances.pred_points is None:
        raise ValueError(
            "DirectionalDistances carries no cleaned prediction points; error-colored "
            "output needs distances from compute_directional_distances()."
        )
    colors = error_colors(distances.pred_to_gt, vmax=vmax)
    pointcloud_backend.save_pointcloud(distances.pred_points, Path(path), colors=colors)


def write_geometry_debug_outputs(
    debug_scenes: list[tuple[str, DirectionalDistances]],
    run_dir: Path,
    *,
    reporting: ReportingSpec,
    pointcloud_backend: Any,
) -> list[dict[str, Any]]:
    """Write requested debug outputs for every captured scene into ``debug/``.

    Returns the manifest records also written to ``debug/debug_outputs.json``.
    A no-op (writes nothing, returns ``[]``) when no output kind is requested or
    no scene captured distances.
    """
    if not (reporting.save_colored_errors or reporting.save_distance_histogram):
        return []
    if not debug_scenes:
        return []

    debug_dir = Path(run_dir) / "debug"
    records: list[dict[str, Any]] = []
    for scene_id, distances in debug_scenes:
        record: dict[str, Any] = {
            "scene_id": scene_id,
            "n_points_pred": distances.n_points_pred,
            "n_points_gt": distances.n_points_gt,
        }
        if reporting.save_colored_errors:
            vmax = float(distances.pred_to_gt.max()) if distances.pred_to_gt.size else 0.0
            ply_path = debug_dir / f"{scene_id}_error.ply"
            write_error_colored_pointcloud(
                distances, ply_path, pointcloud_backend=pointcloud_backend, vmax=vmax
            )
            record["error_colored_ply"] = ply_path.name
            record["colormap"] = ERROR_COLORMAP
            record["colormap_vmax"] = vmax
        if reporting.save_distance_histogram:
            png_path = debug_dir / f"{scene_id}_histogram.png"
            write_distance_histogram(distances, png_path, scene_id=scene_id)
            record["histogram_png"] = png_path.name
            record["histogram_bins"] = HISTOGRAM_BINS
        records.append(record)

    dump_json({"debug_outputs": records}, debug_dir / "debug_outputs.json")
    return records
