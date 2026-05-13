"""Validate a prediction directory against its manifest + sanity checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.io.trajectory import load_trajectory_tum
from eval3r.manifest._hash import sha256_file
from eval3r.manifest.manifest import (
    MANIFEST_FILENAME,
    Artifact,
    CoordinateSystem,
    Manifest,
    PoseConvention,
    Unit,
)
from eval3r.utils.typing import PathLike


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        if not ok:
            self.errors.append(f"{name}: {detail}" if detail else name)
            self.ok = False

    def warn(self, name: str, detail: str = "") -> None:
        self.checks.append((name, True, f"warning: {detail}" if detail else "warning"))
        self.warnings.append(f"{name}: {detail}" if detail else name)


def _check_artifact(
    art: Artifact, root: Path, label: str, report: ValidationReport
) -> Path | None:
    p = root / art.path
    if not p.exists():
        report.add(f"{label} exists", False, f"{art.path} missing under {root}")
        return None
    actual = sha256_file(p)
    if actual != art.sha256:
        report.add(
            f"{label} sha256",
            False,
            f"{art.path} manifest={art.sha256[:12]}… actual={actual[:12]}…",
        )
        return None
    report.add(f"{label} sha256", True)
    return p


def validate_prediction(path: PathLike) -> ValidationReport:
    root = Path(path)
    report = ValidationReport(ok=True)

    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        report.add("manifest present", False, f"no {MANIFEST_FILENAME} under {root}")
        return report
    report.add("manifest present", True)

    try:
        m = Manifest.model_validate(json.loads(manifest_path.read_text()))
    except Exception as e:
        report.add("manifest parses", False, str(e))
        return report
    report.add("manifest parses", True)

    if m.unit is Unit.UNSPECIFIED:
        report.warn("unit specified", "unit is 'unspecified'")
    if m.coordinate_system is CoordinateSystem.UNSPECIFIED:
        report.warn("coordinate_system specified", "coordinate_system is 'unspecified'")
    if m.pose_convention is PoseConvention.UNSPECIFIED and (
        m.trajectory.tum is not None or m.trajectory.kitti is not None
    ):
        report.warn("pose_convention specified", "pose_convention is 'unspecified'")

    # geometry
    if m.geometry.mesh is not None:
        p = _check_artifact(m.geometry.mesh, root, "mesh", report)
        if p is not None:
            try:
                mesh = load_mesh(p)
                ok = np.isfinite(mesh.vertices).all() and len(mesh.vertices) > 0
                report.add("mesh finite & nonempty", ok)
            except Exception as e:
                report.add("mesh loads", False, str(e))

    if m.geometry.point_cloud is not None:
        p = _check_artifact(m.geometry.point_cloud, root, "point_cloud", report)
        if p is not None:
            try:
                pc = load_point_cloud(p)
                ok = np.isfinite(pc.points).all() and len(pc.points) > 0
                report.add("points finite & nonempty", ok)
            except Exception as e:
                report.add("point_cloud loads", False, str(e))

    # trajectory
    if m.trajectory.tum is not None:
        p = _check_artifact(m.trajectory.tum, root, "trajectory.tum", report)
        if p is not None:
            try:
                traj = load_trajectory_tum(p)
                if traj.timestamps is not None:
                    report.add(
                        "tum poses match timestamps",
                        traj.timestamps.shape[0] == traj.poses.shape[0],
                    )
            except Exception as e:
                report.add("trajectory.tum loads", False, str(e))

    if m.trajectory.kitti is not None:
        _check_artifact(m.trajectory.kitti, root, "trajectory.kitti", report)

    # cameras
    if m.cameras.intrinsics is not None:
        _check_artifact(m.cameras.intrinsics, root, "cameras.intrinsics", report)

    return report
