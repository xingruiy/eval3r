"""ETH3D official multi-view-evaluation wrapper (``official_eval`` kind).

This backend does **not** reimplement the ETH3D evaluation. It wraps the official
``ETH3D/multi-view-evaluation`` C++ tool, which computes accuracy / completeness /
F1 at a list of tolerances against the laser-scan ground truth referenced by a
MeshLab project (``scan_alignment.mlp``). The official scoring is not a plain
inlier fraction: completeness and accuracy are normalized per voxel cell over two
shifted voxel grids, and accuracy classifies reconstruction points as accurate /
inaccurate / unobserved using beam-based free-space modeling from the scanner
positions. None of that is reproducible with plain distance metrics, so eval3r
only resolves the per-scene inputs, invokes the official binary as a subprocess,
and parses its printed summary (``official_script_wrapper``).

The official tool is a **user-supplied external build** (like the Tanks and
Temples toolbox checkout and the DTU MATLAB path: not a pip package). It is
located from an explicit ``tool_path`` or the ``EVAL3R_ETH3D_TOOL`` /
``ETH3D_MULTI_VIEW_EVALUATION`` environment variable. When absent, the wrapper
raises an explicit error naming what is missing and where to get it, so a run
fails loudly rather than emitting unofficial numbers, and wrapper tests skip
cleanly. ``voxel_size`` and the beam free-space parameters are left at the
official defaults unless explicitly overridden; whatever is in effect is recorded
in the result metadata.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendError, BackendInfo

_TOOL_ENV_VARS = ("EVAL3R_ETH3D_TOOL", "ETH3D_MULTI_VIEW_EVALUATION")
_TOOL_URL = "https://github.com/ETH3D/multi-view-evaluation (build ETH3DMultiViewEvaluation)"

# Official defaults from the tool's --help / source (main.cc). Recorded per scene.
OFFICIAL_VOXEL_SIZE = 0.01
OFFICIAL_BEAM_START_RADIUS_METERS = 0.5 * 0.00225
OFFICIAL_BEAM_DIVERGENCE_HALFANGLE_DEG = 0.011

# The tool prints four aligned lines, e.g. ``Tolerances: 0.01 0.02`` etc.
_OUTPUT_LINES = {
    "tolerances": re.compile(r"^Tolerances:\s*(.*)$", re.MULTILINE),
    "completenesses": re.compile(r"^Completenesses:\s*(.*)$", re.MULTILINE),
    "accuracies": re.compile(r"^Accuracies:\s*(.*)$", re.MULTILINE),
    "f1_scores": re.compile(r"^F1-scores:\s*(.*)$", re.MULTILINE),
}


@dataclass(frozen=True)
class Eth3dEvalResult:
    """Official ETH3D per-scene scores (per tolerance) plus provenance."""

    tolerances: list[float]
    accuracies: dict[float, float]
    completenesses: dict[float, float]
    f1_scores: dict[float, float]
    command: list[str] = field(default_factory=list)
    tool_path: str | None = None
    tool_commit: str | None = None
    voxel_size: float = OFFICIAL_VOXEL_SIZE
    beam_start_radius_meters: float = OFFICIAL_BEAM_START_RADIUS_METERS
    beam_divergence_halfangle_deg: float = OFFICIAL_BEAM_DIVERGENCE_HALFANGLE_DEG


def parse_official_output(text: str, requested_tolerances: list[float]) -> dict[str, list[float]]:
    """Parse the four summary lines of the official tool.

    Pure and independently testable. Every requested tolerance must appear with a
    completeness, accuracy, and F1 value; anything missing or misaligned is a
    :class:`MetricError` (an incomplete official run must not be scored).
    """
    parsed: dict[str, list[float]] = {}
    missing: list[str] = []
    for key, pattern in _OUTPUT_LINES.items():
        match = pattern.search(text)
        if match is None:
            missing.append(key)
            continue
        try:
            parsed[key] = [float(v) for v in match.group(1).split()]
        except ValueError as exc:
            raise MetricError(
                f"could not parse the ETH3D official output line for {key}: "
                f"{match.group(1)!r}. Raw output:\n{text[-2000:]}"
            ) from exc
    if missing:
        raise MetricError(
            "could not parse the ETH3D official output; missing line(s): "
            f"{', '.join(missing)}. The official tool must print Tolerances, "
            f"Completenesses, Accuracies, and F1-scores. Raw output:\n{text[-2000:]}"
        )

    n = len(parsed["tolerances"])
    for key in ("completenesses", "accuracies", "f1_scores"):
        if len(parsed[key]) != n:
            raise MetricError(
                f"the ETH3D official output is misaligned: {n} tolerances but "
                f"{len(parsed[key])} {key} values. Raw output:\n{text[-2000:]}"
            )
    for tol in requested_tolerances:
        if not any(abs(tol - got) <= 1e-9 for got in parsed["tolerances"]):
            raise MetricError(
                f"the ETH3D official output reports tolerances {parsed['tolerances']} but the "
                f"protocol requested tolerance {tol}; refusing to score from a mismatched run."
            )
    return parsed


class Eth3dOfficialEval:
    """Wrapper around the official ETH3D multi-view-evaluation binary."""

    name = "eth3d_official"
    method = "official_script_wrapper"
    # The generic benchmark loop uses this to route the scan-MLP official path
    # (reconstruction PLY + scan_alignment.mlp) rather than the DTU point-array or
    # Tanks-and-Temples artifacts paths.
    input_mode = "scan_mlp"

    def __init__(self, tool_path: Path | str | None = None) -> None:
        self.tool_path = Path(tool_path) if tool_path is not None else None

    # --- tool resolution ---------------------------------------------------------

    def _resolve_tool(self) -> Path:
        candidate = self.tool_path
        if candidate is None:
            for var in _TOOL_ENV_VARS:
                value = os.environ.get(var)
                if value:
                    candidate = Path(value)
                    break
        if candidate is None:
            raise BackendError(
                "the ETH3D official evaluation tool is not configured. It is a user-supplied "
                f"external build of {_TOOL_URL}. Set one of {', '.join(_TOOL_ENV_VARS)} to the "
                "ETH3DMultiViewEvaluation binary path, or pass tool_path."
            )
        candidate = Path(candidate)
        if not candidate.is_file():
            raise BackendError(
                f"the ETH3D official evaluation binary does not exist at {candidate}. Point "
                f"{_TOOL_ENV_VARS[0]} at a built ETH3DMultiViewEvaluation from {_TOOL_URL}."
            )
        return candidate

    def _tool_commit(self, tool_path: Path) -> str | None:
        """Commit of the tool's source checkout when the binary lives inside one."""
        try:
            out = subprocess.run(
                ["git", "-C", str(tool_path.parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        return commit or None

    def backend_info(self) -> BackendInfo:
        commit: str | None = None
        library = "ETH3D-multi-view-evaluation(subprocess)"
        try:
            tool = self._resolve_tool()
            commit = self._tool_commit(tool)
            library = f"ETH3D-multi-view-evaluation(subprocess, {tool})"
        except BackendError:
            # backend_info must never crash metadata assembly; record it as unresolved.
            pass
        return BackendInfo(
            kind="official_eval",
            name=self.name,
            library=library,
            version=commit or "external_build",
            approximate=False,
        )

    # --- evaluation ----------------------------------------------------------------

    def evaluate_scene(
        self,
        scene_id: str,
        *,
        scan_mlp_path: Path,
        ply_path: Path,
        tolerances: list[float],
        timeout: float = 3600.0,
    ) -> Eth3dEvalResult:
        """Run the official tool for one scene and parse its summary.

        ``scan_mlp_path`` is the scene's ``scan_alignment.mlp`` (which references the
        laser-scan PLYs and their poses); ``ply_path`` is the prediction point cloud;
        ``tolerances`` is the protocol's tolerance list in metres.
        """
        if not tolerances:
            raise MetricError(
                f"no evaluation tolerances were provided for ETH3D scene '{scene_id}'; the "
                f"protocol must define the official tolerance set explicitly."
            )
        tool = self._resolve_tool()
        sorted_tolerances = sorted(tolerances)
        command = [
            str(tool),
            "--reconstruction_ply_path", str(ply_path),
            "--ground_truth_mlp_path", str(scan_mlp_path),
            "--tolerances", ",".join(repr(t) for t in sorted_tolerances),
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise MetricError(
                f"the ETH3D official tool timed out after {timeout}s for scene '{scene_id}'. "
                f"Command: {' '.join(command)}"
            ) from exc
        except OSError as exc:
            raise MetricError(
                f"failed to launch the ETH3D official tool for scene '{scene_id}': {exc}. "
                f"Command: {' '.join(command)}"
            ) from exc

        if proc.returncode != 0:
            raise MetricError(
                f"the ETH3D official tool failed for scene '{scene_id}' "
                f"(exit {proc.returncode}). Command: {' '.join(command)}\n"
                f"stderr:\n{proc.stderr[-2000:]}\nstdout:\n{proc.stdout[-2000:]}"
            )

        parsed = parse_official_output(proc.stdout + "\n" + proc.stderr, sorted_tolerances)
        reported = parsed["tolerances"]
        return Eth3dEvalResult(
            tolerances=reported,
            accuracies=dict(zip(reported, parsed["accuracies"], strict=True)),
            completenesses=dict(zip(reported, parsed["completenesses"], strict=True)),
            f1_scores=dict(zip(reported, parsed["f1_scores"], strict=True)),
            command=command,
            tool_path=str(tool),
            tool_commit=self._tool_commit(tool),
        )

    @staticmethod
    def lookup(values: dict[float, float], tolerance: float) -> float:
        """Value at ``tolerance``, tolerant to float32 rounding in the tool's echo."""
        for got, value in values.items():
            if abs(got - tolerance) <= 1e-9 + 1e-6 * abs(tolerance):
                return value
        raise MetricError(
            f"the ETH3D official output has no value at tolerance {tolerance}; "
            f"reported tolerances: {sorted(values)}."
        )
