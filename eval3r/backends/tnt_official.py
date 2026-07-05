"""Tanks and Temples official evaluation wrapper (``official_eval`` kind).

This backend does **not** reimplement the Tanks and Temples evaluation. It wraps the
official Python toolbox (``isl-org/TanksAndTemples`` ``python_toolbox/evaluation``),
which performs trajectory-based alignment, ICP refinement, crop-volume masking, and
precision/recall/F-score at the official per-scene distance threshold ``dTau``. eval3r
only resolves per-scene artifacts, invokes the official ``run.py`` as a subprocess, and
parses its printed summary.

The official toolbox is a **user-supplied external checkout** (like the DTU MATLAB
path: not a pip package). It is located from an explicit ``toolbox_dir`` or the
``EVAL3R_TNT_TOOLBOX`` / ``TANKSANDTEMPLES_TOOLBOX`` environment variable. When it is
absent the wrapper raises an explicit error naming what is missing and where to get it,
so a run fails loudly rather than emitting unofficial numbers, and wrapper tests skip
cleanly. The per-scene threshold is read from the official output, never hardcoded here.

The toolbox pins ``open3d==0.9`` (its ``requirements.txt``), whose ``open3d.registration``
API and RANSAC convergence semantics differ from newer open3d in ways that could change
the alignment and therefore the score. Rather than porting the official code to a newer
open3d (a result-affecting change), the toolbox is run **unmodified** under its own pinned
interpreter, configured via an explicit ``python_executable`` or the ``EVAL3R_TNT_PYTHON``
environment variable (default: the current interpreter). The interpreter used is recorded
in result metadata so the environment behind the official numbers is auditable.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendError, BackendInfo

_TOOLBOX_ENV_VARS = ("EVAL3R_TNT_TOOLBOX", "TANKSANDTEMPLES_TOOLBOX")
# The toolbox pins open3d==0.9; run it under its own interpreter, not necessarily ours.
_PYTHON_ENV_VARS = ("EVAL3R_TNT_PYTHON",)
_TOOLBOX_URL = "https://github.com/isl-org/TanksAndTemples (python_toolbox/evaluation)"

# The official run.py prints lines like ``precision : 0.9033`` / ``distance tau : 0.003``.
_METRIC_PATTERNS = {
    "precision": re.compile(r"precision\s*[:=]\s*([-+0-9.eE]+)", re.IGNORECASE),
    "recall": re.compile(r"recall\s*[:=]\s*([-+0-9.eE]+)", re.IGNORECASE),
    "fscore": re.compile(r"f-?score\s*[:=]\s*([-+0-9.eE]+)", re.IGNORECASE),
    "distance_tau": re.compile(r"distance\s*tau\s*[:=]\s*([-+0-9.eE]+)", re.IGNORECASE),
}


@dataclass(frozen=True)
class TntEvalResult:
    """Official Tanks and Temples scores for one scene plus provenance."""

    precision: float
    recall: float
    fscore: float
    distance_tau: float
    command: list[str] = field(default_factory=list)
    toolbox_dir: str | None = None
    toolbox_commit: str | None = None
    python_executable: str | None = None
    out_dir: str | None = None


def parse_official_output(text: str) -> dict[str, float]:
    """Parse precision / recall / f-score / distance_tau from the official summary.

    Pure and independently testable: the official ``run.py`` prints these four labelled
    values. Missing any of them is a :class:`MetricError` (the official run did not
    produce a complete summary, so a score cannot be claimed).
    """
    values: dict[str, float] = {}
    missing: list[str] = []
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            missing.append(key)
            continue
        values[key] = float(match.group(1))
    if missing:
        raise MetricError(
            "could not parse the Tanks and Temples official output; missing "
            f"{', '.join(missing)}. The official run.py summary must report precision, recall, "
            f"f-score, and distance tau. Raw output:\n{text[-2000:]}"
        )
    return values


class TntOfficialEval:
    """Wrapper around the official Tanks and Temples evaluation toolbox."""

    name = "tnt_official"
    method = "official_script_wrapper"
    # The generic benchmark loop uses this to route the file-based (artifacts) official
    # path rather than the DTU-style point-array path.
    input_mode = "artifacts"

    def __init__(
        self,
        toolbox_dir: Path | str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.toolbox_dir = Path(toolbox_dir) if toolbox_dir is not None else None
        self.python_executable = python_executable

    # --- toolbox resolution ----------------------------------------------------

    def _resolve_python(self) -> str:
        """Interpreter that runs the toolbox (its pinned open3d==0.9 env when configured).

        Explicit ``python_executable`` wins, then ``EVAL3R_TNT_PYTHON``; otherwise the
        current interpreter. Recorded in metadata so the env behind the numbers is known.
        """
        if self.python_executable:
            return self.python_executable
        for var in _PYTHON_ENV_VARS:
            value = os.environ.get(var)
            if value:
                return value
        return sys.executable

    def _resolve_toolbox(self) -> Path:
        candidate = self.toolbox_dir
        if candidate is None:
            for var in _TOOLBOX_ENV_VARS:
                value = os.environ.get(var)
                if value:
                    candidate = Path(value)
                    break
        if candidate is None:
            raise BackendError(
                "the Tanks and Temples official toolbox is not configured. It is a user-supplied "
                f"external checkout of {_TOOLBOX_URL}. Set one of {', '.join(_TOOLBOX_ENV_VARS)} "
                "to its evaluation directory (containing run.py), or pass toolbox_dir."
            )
        candidate = Path(candidate)
        run_py = candidate / "run.py"
        if not run_py.is_file():
            raise BackendError(
                f"the Tanks and Temples toolbox at {candidate} has no run.py. Point "
                f"{_TOOLBOX_ENV_VARS[0]} at the official {_TOOLBOX_URL} evaluation directory."
            )
        return candidate

    def _toolbox_commit(self, toolbox_dir: Path) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(toolbox_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = out.stdout.strip()
        return commit or None

    def backend_info(self) -> BackendInfo:
        commit: str | None = None
        library = "TanksAndTemples-toolbox(subprocess)"
        try:
            toolbox = self._resolve_toolbox()
            commit = self._toolbox_commit(toolbox)
            library = f"TanksAndTemples-toolbox(subprocess, {toolbox})"
        except BackendError:
            # backend_info must never crash metadata assembly; record it as unresolved.
            pass
        return BackendInfo(
            kind="official_eval",
            name=self.name,
            library=library,
            version=commit or "external_checkout",
            approximate=False,
        )

    # --- evaluation ------------------------------------------------------------

    def evaluate_scene(
        self,
        scene_id: str,
        *,
        dataset_dir: Path,
        traj_path: Path,
        ply_path: Path,
        out_dir: Path,
        timeout: float = 3600.0,
    ) -> TntEvalResult:
        """Run the official toolbox for one scene and parse its summary.

        ``dataset_dir`` is the scene directory the toolbox reads GT/crop/trans/mapping
        from by naming convention; ``traj_path`` is the ``.log`` trajectory; ``ply_path``
        is the prediction point cloud; ``out_dir`` receives the toolbox's output files.
        """
        toolbox = self._resolve_toolbox()
        python_executable = self._resolve_python()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            python_executable,
            str(toolbox / "run.py"),
            "--dataset-dir", str(dataset_dir),
            "--traj-path", str(traj_path),
            "--ply-path", str(ply_path),
            "--out-dir", str(out_dir),
        ]
        try:
            proc = subprocess.run(
                command, cwd=str(toolbox), capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise MetricError(
                f"the Tanks and Temples official toolbox timed out after {timeout}s for scene "
                f"'{scene_id}'. Command: {' '.join(command)}"
            ) from exc
        except OSError as exc:
            raise MetricError(
                f"failed to launch the Tanks and Temples official toolbox for scene '{scene_id}': "
                f"{exc}. Command: {' '.join(command)}"
            ) from exc

        if proc.returncode != 0:
            raise MetricError(
                f"the Tanks and Temples official toolbox failed for scene '{scene_id}' "
                f"(exit {proc.returncode}). Command: {' '.join(command)}\n"
                f"stderr:\n{proc.stderr[-2000:]}\nstdout:\n{proc.stdout[-2000:]}"
            )

        values = parse_official_output(proc.stdout + "\n" + proc.stderr)
        return TntEvalResult(
            precision=values["precision"],
            recall=values["recall"],
            fscore=values["fscore"],
            distance_tau=values["distance_tau"],
            command=command,
            toolbox_dir=str(toolbox),
            toolbox_commit=self._toolbox_commit(toolbox),
            python_executable=python_executable,
            out_dir=str(out_dir),
        )
