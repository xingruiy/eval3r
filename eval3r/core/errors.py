"""Explicit eval3r exception types with rich, actionable messages.

Per CLAUDE.md, handled errors must state what failed, why, and the concrete inputs
involved (file path, protocol name, the field/check that failed). These exception
types carry that context rather than reducing failures to booleans or bare exits.
"""

from __future__ import annotations


class Eval3rError(Exception):
    """Base class for all eval3r errors."""


class InvalidGeometryError(Eval3rError):
    """A point/mesh array had the wrong shape, was empty, or held non-finite values."""


class InvalidDepthError(Eval3rError):
    """A depth array could not be loaded/validated, or masking left no valid pixels."""


class InvalidTrajectoryError(Eval3rError):
    """A trajectory could not be loaded, or timestamp association found no matches."""


class MetricError(Eval3rError):
    """A metric could not be computed (unsupported name, missing required spec field)."""


class AlignmentError(Eval3rError):
    """An alignment transform could not be estimated or is disallowed by the protocol."""


class PoseConventionError(Eval3rError):
    """A pose/coordinate-convention transform could not be applied or failed validation.

    Raised by the convention transformer (``eval3r/core/pose_convention.py``) and the
    source-format mapping (``eval3r/datasets/conventions.py``) when a source format
    cannot be mapped to a verified convention, or when a produced transform fails a
    truthfulness invariant (non-orthonormal/reflected rotation, broken homogeneous row,
    non-finite value, failed round-trip, or a camera-center/inverse-consistency check).
    Messages name the pose index, the invariant, the numeric residual versus tolerance,
    and the ``(src -> dst)`` conversion involved.
    """


class CullingError(Eval3rError):
    """A masking/culling policy could not be applied on the single-file path."""


class SceneEvaluationError(Eval3rError):
    """A scene failed under an ``abort`` failure policy.

    Carries the scene id and the pipeline stage that failed so the runner and CLI
    can report exactly what failed without reducing it to a bare exit code.
    """

    def __init__(self, scene_id: str, stage: str, reason: str) -> None:
        self.scene_id = scene_id
        self.stage = stage
        self.reason = reason
        super().__init__(
            f"scene '{scene_id}' failed at stage '{stage}' and the failure policy is "
            f"'abort': {reason}"
        )


class RunComparisonError(Eval3rError):
    """Two run results cannot be compared under the requested diff mode.

    Raised by strict ``diff_runs`` when protocol hashes differ (metric numbers under
    different protocols are not comparable), or when a run directory is missing or
    unreadable. The message names both runs and, for hash mismatches, how to request
    an explicitly-labeled non-strict comparison instead.
    """


class DatasetError(Eval3rError):
    """A dataset adapter could not resolve scenes, ground truth, or predictions."""


class UnknownDatasetError(DatasetError):
    """A dataset adapter name is not registered."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        avail = ", ".join(sorted(available)) if available else "(none registered)"
        super().__init__(
            f"no dataset adapter named '{name}' is registered. Available adapters: {avail}."
        )


class BenchmarkError(Eval3rError):
    """A benchmark run was refused before computation (capability/local-eval gating)."""


class PredictionLayoutError(Eval3rError):
    """A prediction directory does not satisfy the eval3r-native layout.

    Raised by the prediction writer/reader (task 019) when a manifest fails schema
    validation, a declared per-scene file is missing, an entry is inconsistent with
    the declared modality, or a recorded fingerprint no longer matches the file on
    disk. Messages name the prediction root, the scene, the entry field, and the
    check that failed.
    """


class ProtocolError(Eval3rError):
    """A protocol could not be loaded, found, or validated."""


class ProtocolNotFoundError(ProtocolError):
    """A protocol name did not resolve to a known built-in or file."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        known_txt = ", ".join(sorted(known)) if known else "(none registered)"
        super().__init__(
            f"protocol '{name}' is not a known built-in protocol. "
            f"Known protocols: {known_txt}. "
            f"Pass an exact built-in name, or a path to a protocol YAML file."
        )


class ProtocolValidationError(ProtocolError):
    """A protocol YAML failed schema validation.

    Wraps the underlying Pydantic error with the source path and protocol name so
    the message points at the file and field that failed.
    """

    def __init__(self, source: str, detail: str, *, name: str | None = None) -> None:
        self.source = source
        self.detail = detail
        self.name = name
        named = f" (protocol '{name}')" if name else ""
        super().__init__(f"protocol file failed validation: {source}{named}\n{detail}")
