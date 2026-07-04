"""Mask stage: masking and culling.

The single-file geometry path supports only ``method: none`` (no dataset masks are
available without an adapter). Any other culling method — observability masks,
visibility culling, scene bounds, official masks — is resolved by dataset adapters
(task 011+). Requesting one here fails explicitly rather than silently evaluating on
unmasked geometry, which would misrepresent the protocol (``.agent/plan.md`` masking
policy: "It should not silently fall back to no mask").
"""

from __future__ import annotations

from eval3r.core.errors import CullingError
from eval3r.core.schema import CullingSpec
from eval3r.pipeline.stages.load import LoadedGeometry


def apply_culling(geometry: LoadedGeometry, spec: CullingSpec, *, role: str) -> LoadedGeometry:
    """Return ``geometry`` unchanged for ``method: none``; otherwise fail explicitly."""
    if spec.method == "none":
        return geometry
    raise CullingError(
        f"{role} culling method '{spec.method}' requires a dataset adapter to resolve the mask "
        f"(observability, visibility, scene bounds, or official masks are provided per dataset, "
        f"task 011+). The single-file geometry path only supports 'none'. Run this comparison "
        f"through 'e3r benchmark run' with a dataset that supplies the mask."
    )
