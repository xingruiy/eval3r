"""Public Python API entry points.

``evaluate_geometry`` runs the single-file geometry pipeline (task 007) for one
prediction/ground-truth file pair under a named or file protocol, optionally writing
a complete run directory. Benchmark and diff entry points are wired up in later
slices (tasks 008 / 016).
"""

from __future__ import annotations

from pathlib import Path

from eval3r.core.environment import capture_environment
from eval3r.core.registry import BackendRegistry
from eval3r.core.result import RunResult
from eval3r.datasets import default_registry as default_dataset_registry
from eval3r.datasets.registry import DatasetRegistry
from eval3r.pipeline.benchmark import BenchmarkRunOutput, run_benchmark_geometry
from eval3r.pipeline.depth_runner import DepthRunOutput, run_single_file_depth
from eval3r.pipeline.pose_runner import PoseRunOutput, run_single_file_pose
from eval3r.pipeline.runner import GeometryRunOutput, run_single_file_geometry
from eval3r.pipeline.stages.load import GeometryKind
from eval3r.protocols import load_protocol
from eval3r.reports.run_directory import write_run_directory


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

    return run if return_run else run.result
