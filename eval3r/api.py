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
