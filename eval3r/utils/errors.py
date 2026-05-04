"""eval3r exception hierarchy and warnings."""

from __future__ import annotations


class Eval3rError(Exception):
    """Base class for all eval3r errors."""


class MissingArtifactError(Eval3rError, FileNotFoundError):
    """A referenced prediction artifact is missing on disk."""


class CorruptedArtifactError(Eval3rError):
    """A prediction artifact's hash does not match the manifest."""


class EmptyGeometryError(Eval3rError, ValueError):
    """Geometry input contains zero vertices/points."""


class NaNGeometryError(Eval3rError, ValueError):
    """Geometry input contains NaN or Inf values."""


class ManifestError(Eval3rError, ValueError):
    """The prediction manifest is invalid or inconsistent."""


class AlignmentError(Eval3rError, ValueError):
    """Alignment failed (e.g., degenerate point sets)."""


class MissingOptionalDependencyError(Eval3rError, ImportError):
    """A feature was requested whose optional dependency is not installed."""


class EvalAssumptionWarning(UserWarning):
    """An evaluation assumption (unit, convention, etc.) was left unspecified."""
