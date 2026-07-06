"""Public Python API entry points.

``evaluate_geometry`` runs the single-file geometry pipeline (task 007) for one
prediction/ground-truth file pair under a named or file protocol, optionally writing
a complete run directory. ``run_benchmark`` evaluates a dataset split (task 008);
``diff_runs`` compares two written run directories (task 016); ``align_geometries``
estimates a standalone SE3/Sim3 alignment with mandatory visualization (task 018).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from eval3r.core.environment import capture_environment
from eval3r.core.errors import AlignmentError
from eval3r.core.protocol import EvalProtocol
from eval3r.core.registry import BackendRegistry, default_registry
from eval3r.core.result import RunResult
from eval3r.core.schema import AlignmentSpec
from eval3r.datasets import default_registry as default_dataset_registry
from eval3r.datasets.registry import DatasetRegistry
from eval3r.pipeline.benchmark import BenchmarkRunOutput, run_benchmark_geometry
from eval3r.pipeline.depth_runner import DepthRunOutput, run_single_file_depth
from eval3r.pipeline.pose_runner import PoseRunOutput, run_single_file_pose
from eval3r.pipeline.runner import GeometryRunOutput, run_single_file_geometry
from eval3r.pipeline.stages.align import (
    DEFAULT_ICP_MAX_ITERATIONS,
    DEFAULT_ICP_MAX_POINTS,
    align_geometry,
    capture_alignment_vis,
)
from eval3r.pipeline.stages.load import GeometryKind, load_geometry
from eval3r.predictions import (  # noqa: F401  (public API re-export)
    PredictionWriter,
    read_prediction_dir,
)
from eval3r.protocols import load_protocol
from eval3r.reports.alignment_vis import (
    write_alignment_vis_outputs,
    write_alignment_visualization,
)
from eval3r.reports.diff import RunDiff, diff_runs  # noqa: F401  (public API re-export)
from eval3r.reports.json import dump_json
from eval3r.reports.plots import write_geometry_debug_outputs
from eval3r.reports.run_directory import write_run_directory

#: `align_geometries(max_corr_dist="auto")` rule: this fraction of the GT bbox diagonal.
AUTO_MAX_CORR_DIST_FRACTION = 0.05


def _pointcloud_backend(protocol: EvalProtocol, registry: BackendRegistry | None) -> Any:
    registry = registry or default_registry()
    return registry.require(
        "pointcloud", protocol.backend_preferences.get("pointcloud", "plyfile")
    )


def _write_debug_outputs(
    debug_scenes: list[tuple[str, Any]],
    protocol: EvalProtocol,
    out_dir: Path,
    registry: BackendRegistry | None,
) -> None:
    """Write protocol-requested debug outputs (colored PLY / histogram) to ``debug/``."""
    if not debug_scenes:
        return
    write_geometry_debug_outputs(
        debug_scenes, out_dir,
        reporting=protocol.reporting,
        pointcloud_backend=_pointcloud_backend(protocol, registry),
    )


def _write_alignment_vis(
    captures: list[Any],
    protocol: EvalProtocol,
    out_dir: Path,
    registry: BackendRegistry | None,
) -> None:
    """Write the mandatory alignment overlays to ``debug/`` when alignment ran."""
    if not captures:
        return
    write_alignment_vis_outputs(
        captures, out_dir, pointcloud_backend=_pointcloud_backend(protocol, registry)
    )


def evaluate_geometry(
    pred: str | Path,
    gt: str | Path,
    *,
    input_type: GeometryKind = "pointcloud",
    gt_type: GeometryKind = "pointcloud",
    threshold: float | None = None,
    sample: int | None = None,
    protocol: str = "single_geometry",
    method: str | None = None,
    out_dir: str | Path | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    return_run: bool = False,
) -> RunResult | GeometryRunOutput:
    """Evaluate one predicted mesh/point cloud against ground-truth geometry.

    ``protocol`` is a built-in name or a path to a protocol YAML. CLI-style
    ``threshold`` / ``sample`` / input-type flags are applied as recorded overrides.
    When ``out_dir`` is given a full run directory is written there. Returns the
    :class:`RunResult` by default, or the richer :class:`GeometryRunOutput` when
    ``return_run`` is true.
    """
    proto = load_protocol(protocol)
    environment = capture_environment(command=command)
    run = run_single_file_geometry(
        pred, gt, proto,
        input_type=input_type, gt_type=gt_type,
        threshold=threshold, sample=sample, method=method,
        registry=registry, command=command, environment=environment,
    )

    if out_dir is not None:
        write_run_directory(
            run.result,
            Path(out_dir),
            protocol=run.protocol,
            config=run.config,
            environment=run.result.environment,
            backend_versions=run.result.backend_versions,
            alignment_transforms=run.alignment_transforms,
        )
        _write_debug_outputs(run.debug_scenes, run.protocol, Path(out_dir), registry)
        _write_alignment_vis(run.alignment_vis, run.protocol, Path(out_dir), registry)

    return run if return_run else run.result


def evaluate_depth(
    pred: str | Path,
    gt: str | Path,
    *,
    depth_unit: float | None = None,
    gt_depth_unit: float | None = None,
    align: str | None = None,
    align_granularity: str | None = None,
    protocol: str = "single_depth",
    method: str | None = None,
    out_dir: str | Path | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    return_run: bool = False,
) -> RunResult | DepthRunOutput:
    """Evaluate a predicted depth map or frame directory against ground-truth depth.

    ``pred``/``gt`` are either two depth files (single frame) or two directories of
    frames matched by filename stem (depth sequence, aggregated per frame — never
    fused into scene geometry). ``depth_unit`` / ``gt_depth_unit`` are metres per
    stored unit and are required for integer depth files. ``align`` /
    ``align_granularity`` override the protocol's scale alignment and are recorded
    as overrides (the protocol hash changes accordingly).
    """
    proto = load_protocol(protocol)
    environment = capture_environment(command=command)
    run = run_single_file_depth(
        pred, gt, proto,
        pred_depth_unit=depth_unit, gt_depth_unit=gt_depth_unit,
        align=align, align_granularity=align_granularity,
        method=method, registry=registry, command=command, environment=environment,
    )

    if out_dir is not None:
        write_run_directory(
            run.result,
            Path(out_dir),
            protocol=run.protocol,
            config=run.config,
            environment=run.result.environment,
            backend_versions=run.result.backend_versions,
            alignment_transforms=run.alignment_records,
        )

    return run if return_run else run.result


def evaluate_pose(
    pred: str | Path,
    gt: str | Path,
    *,
    align: str | None = None,
    associate_max_diff: float | None = None,
    backend: str | None = None,
    protocol: str = "single_pose",
    method: str | None = None,
    out_dir: str | Path | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    return_run: bool = False,
) -> RunResult | PoseRunOutput:
    """Evaluate a predicted trajectory against a ground-truth trajectory.

    ``pred``/``gt`` are TUM-format trajectory files (``timestamp x y z qx qy qz
    qw``). ``align`` overrides the protocol's trajectory alignment (``none`` /
    ``se3`` / ``sim3``, or the full ``trajectory_se3`` / ``trajectory_sim3``
    names) and ``associate_max_diff`` the timestamp-association tolerance in
    seconds; both are recorded as overrides (the protocol hash changes
    accordingly). Association counts, alignment mode, and the estimated Sim3
    scale are recorded in result metadata and ``alignment_transforms.json``.
    """
    proto = load_protocol(protocol)
    environment = capture_environment(command=command)
    run = run_single_file_pose(
        pred, gt, proto,
        align=align, associate_max_diff=associate_max_diff, backend=backend,
        method=method, registry=registry, command=command, environment=environment,
    )

    if out_dir is not None:
        write_run_directory(
            run.result,
            Path(out_dir),
            protocol=run.protocol,
            config=run.config,
            environment=run.result.environment,
            backend_versions=run.result.backend_versions,
            alignment_transforms=run.alignment_records,
        )

    return run if return_run else run.result


@dataclass
class AlignmentOutput:
    """Everything ``align_geometries`` produced: the transform and the artifact paths."""

    alignment: dict[str, Any]
    out_dir: Path
    alignment_json: Path
    pred_aligned: Path
    vis_manifest: dict[str, Any]


def align_geometries(
    pred: str | Path,
    gt: str | Path,
    *,
    out_dir: str | Path,
    mode: str = "se3",
    solver: str = "icp",
    input_type: GeometryKind = "pointcloud",
    gt_type: GeometryKind = "pointcloud",
    max_corr_dist: float | str | None = None,
    max_iterations: int = DEFAULT_ICP_MAX_ITERATIONS,
    max_points: int = DEFAULT_ICP_MAX_POINTS,
    pred_trajectory: str | Path | None = None,
    gt_trajectory: str | Path | None = None,
    associate_max_diff: float | None = None,
    registry: BackendRegistry | None = None,
) -> AlignmentOutput:
    """Estimate an SE3/Sim3 transform mapping ``pred`` onto ``gt`` — and show it.

    Two solvers only (no feature-based global registration): ``"icp"`` runs
    closest-point ICP on the geometries (``max_corr_dist`` required — a number in
    metres, or ``"auto"`` for 5% of the GT bounding-box diagonal, resolved and
    recorded); ``"trajectory"`` aligns the predicted camera trajectory onto the GT
    trajectory (TUM files, evo association + Umeyama; ``associate_max_diff``
    required) and propagates that transform to the geometry.

    The visualization is mandatory: ``out_dir`` always receives
    ``alignment_before.ply`` / ``alignment_after.ply`` overlays, an orthographic
    ``alignment_projections.png``, the ``alignment_vis.json`` manifest, the
    ``pred_aligned.ply`` result, and ``alignment.json`` with the full transform
    provenance. A residual number alone does not show a flipped or locally-stuck
    registration — look at the overlays.
    """
    registry = registry or default_registry()
    pred_path = Path(pred)
    gt_path = Path(gt)
    out_path = Path(out_dir)

    if mode not in ("se3", "sim3"):
        raise AlignmentError(
            f"alignment mode '{mode}' is not supported by align_geometries: use 'se3' "
            f"(rigid) or 'sim3' (with scale) so scale handling stays explicit."
        )
    if solver not in ("icp", "trajectory"):
        raise AlignmentError(
            f"alignment solver '{solver}' is not supported by align_geometries: use "
            f"'icp' (closest-point ICP on the geometries) or 'trajectory' "
            f"(trajectory-first Umeyama propagation). There is deliberately no "
            f"feature-based global registration."
        )

    mesh_backend = registry.require("mesh", "trimesh")
    pc_backend = registry.require("pointcloud", "plyfile")
    pred_geom = load_geometry(
        pred_path, input_type, mesh_backend=mesh_backend, pointcloud_backend=pc_backend
    )
    gt_geom = load_geometry(
        gt_path, gt_type, mesh_backend=mesh_backend, pointcloud_backend=pc_backend
    )

    scene_id = pred_path.stem or "alignment"
    parameters: dict[str, Any] = {}
    if solver == "icp":
        if max_corr_dist is None:
            raise AlignmentError(
                "ICP alignment requires max_corr_dist: pass a distance in metres, or "
                "'auto' to use 5% of the GT bounding-box diagonal (the resolved value "
                "is recorded)."
            )
        if isinstance(max_corr_dist, str):
            if max_corr_dist != "auto":
                raise AlignmentError(
                    f"max_corr_dist must be a number of metres or the string 'auto'; "
                    f"got '{max_corr_dist}'."
                )
            gt_points = gt_geom.alignment_points()
            diagonal = float(
                np.linalg.norm(gt_points.max(axis=0) - gt_points.min(axis=0))
            )
            if diagonal <= 0.0:
                raise AlignmentError(
                    f"cannot resolve max_corr_dist='auto': the GT bounding box of "
                    f"{gt_path} is degenerate (diagonal {diagonal})."
                )
            resolved_mcd = AUTO_MAX_CORR_DIST_FRACTION * diagonal
            parameters["max_corr_dist_auto"] = True
            parameters["max_corr_dist_auto_rule"] = (
                f"{AUTO_MAX_CORR_DIST_FRACTION} * gt_bbox_diagonal ({diagonal:.6g} m)"
            )
        else:
            resolved_mcd = float(max_corr_dist)
        parameters.update(
            {
                "max_correspondence_distance": resolved_mcd,
                "max_iterations": int(max_iterations),
                "max_points": int(max_points),
            }
        )
        spec = AlignmentSpec(
            mode=cast(Literal["se3", "sim3"], mode),
            estimate_on="pointcloud", solver="icp",
            granularity="per_scene", allow_override=True, parameters=parameters,
        )
    else:
        missing = [
            name
            for name, value in (
                ("pred_trajectory", pred_trajectory),
                ("gt_trajectory", gt_trajectory),
                ("associate_max_diff", associate_max_diff),
            )
            if value is None
        ]
        if missing:
            raise AlignmentError(
                f"trajectory-first alignment requires {', '.join(missing)}: both TUM "
                f"trajectory files and an explicit timestamp association tolerance "
                f"in seconds."
            )
        parameters["associate_max_diff"] = float(associate_max_diff)  # type: ignore[arg-type]
        spec = AlignmentSpec(
            mode=cast(Literal["se3", "sim3"], mode),
            estimate_on="trajectory", solver="umeyama",
            granularity="per_scene", allow_override=True, parameters=parameters,
        )

    # metric_scale=False: this is a standalone alignment tool, not a metric-fidelity
    # protocol — Sim3 is exactly what it exists for, and the scale is recorded.
    pred_aligned, result = align_geometry(
        pred_geom, gt_geom, spec,
        scene_id=scene_id, metric_scale=False,
        registration_backend=(
            registry.require("registration", "open3d") if solver == "icp" else None
        ),
        trajectory_backend=(
            registry.require("trajectory", "evo") if solver == "trajectory" else None
        ),
        pred_trajectory=Path(pred_trajectory) if pred_trajectory else None,
        gt_trajectory=Path(gt_trajectory) if gt_trajectory else None,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    vis = capture_alignment_vis(pred_geom, gt_geom, result)
    vis_manifest = write_alignment_visualization(
        vis, out_path, pointcloud_backend=pc_backend
    )
    dump_json({"alignment_visualizations": [vis_manifest]}, out_path / "alignment_vis.json")

    pred_aligned_path = out_path / "pred_aligned.ply"
    if pred_aligned.kind == "mesh":
        mesh_backend.export_mesh(pred_aligned.mesh, pred_aligned_path)
    else:
        pc_backend.save_pointcloud(pred_aligned.points, pred_aligned_path)

    alignment_record = result.as_dict()
    alignment_json = out_path / "alignment.json"
    dump_json(
        {
            "inputs": {
                "pred": str(pred_path),
                "gt": str(gt_path),
                "input_type": input_type,
                "gt_type": gt_type,
                "pred_trajectory": str(pred_trajectory) if pred_trajectory else None,
                "gt_trajectory": str(gt_trajectory) if gt_trajectory else None,
            },
            "requested": {"mode": mode, "solver": solver},
            "alignment": alignment_record,
            "backend_versions": registry.backend_versions(
                {"registration": "open3d"} if solver == "icp" else {"trajectory": "evo"}
            ),
        },
        alignment_json,
    )

    return AlignmentOutput(
        alignment=alignment_record,
        out_dir=out_path,
        alignment_json=alignment_json,
        pred_aligned=pred_aligned_path,
        vis_manifest=vis_manifest,
    )


def run_benchmark(
    pred_root: str | Path,
    *,
    dataset: str,
    split: str,
    protocol: str,
    root: str | Path | None = None,
    manifest: str | Path | None = None,
    method: str | None = None,
    out_dir: str | Path | None = None,
    dataset_registry: DatasetRegistry | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    return_run: bool = False,
) -> RunResult | BenchmarkRunOutput:
    """Evaluate a method's predictions across a dataset split under a named protocol.

    ``dataset`` is a registered adapter name; ``root`` is the dataset root the adapter
    needs (e.g. GT files). The run is refused before computation if the split is not
    locally evaluable. When ``out_dir`` is given a full run directory (including the
    resolved-or-inferred ``manifest.yaml``) is written there.
    """
    proto = load_protocol(protocol)
    dataset_registry = dataset_registry or default_dataset_registry()
    adapter = dataset_registry.create(dataset, Path(root) if root is not None else None)
    environment = capture_environment(command=command)

    run = run_benchmark_geometry(
        pred_root, adapter, proto, split,
        manifest_path=manifest, registry=registry,
        command=command, environment=environment, method=method,
    )

    if out_dir is not None:
        write_run_directory(
            run.result,
            Path(out_dir),
            protocol=run.protocol,
            manifest=run.manifest,
            config=run.config,
            environment=run.result.environment,
            backend_versions=run.result.backend_versions,
            alignment_transforms=run.alignment_transforms,
        )
        _write_debug_outputs(run.debug_scenes, run.protocol, Path(out_dir), registry)
        _write_alignment_vis(run.alignment_vis, run.protocol, Path(out_dir), registry)

    return run if return_run else run.result
