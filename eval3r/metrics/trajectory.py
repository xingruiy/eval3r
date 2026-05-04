"""Trajectory metrics — deferred to v0.2 (requires the [traj] extra)."""

from __future__ import annotations

from eval3r.utils.errors import MissingOptionalDependencyError


def evaluate_trajectory(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise MissingOptionalDependencyError(
        "Trajectory metrics are not implemented in eval3r v0.1. "
        "They will land alongside the [traj] extra in a follow-up release."
    )
