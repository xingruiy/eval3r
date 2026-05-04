"""Normal-consistency metric — deferred to v0.2."""

from __future__ import annotations

from eval3r.utils.errors import MissingOptionalDependencyError


def normal_consistency(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise MissingOptionalDependencyError(
        "Normal-consistency metric is not implemented in eval3r v0.1."
    )
