"""First-class pose / coordinate convention transforms with truthfulness checks.

eval3r normalizes everything to one internal convention — **camera-to-world, OpenCV
axes** (``+X`` right / ``+Y`` down / ``+Z`` forward), metres. A method may instead
emit poses in OpenGL axes (``+X`` right / ``+Y`` up / ``+Z`` backward), or
world-to-camera (``Tcw``) instead of camera-to-world (``Twc``), or geometry built in
an OpenGL *world* frame. This module converts poses **and** geometry between all such
modes and validates every result before it is used, so a convention mismatch is fixed
deterministically rather than left to alignment (which frequently fails to find a 180°
flip) or scored silently in the wrong frame.

The one geometric fact used throughout is that the OpenCV<->OpenGL axis flip is the
single involutive rotation ``F = diag(1, -1, -1, 1)`` — a proper 180 degree rotation
about the camera X axis (``det(F[:3, :3]) = +1``, ``F @ F = I``). The **same** ``F``
serves two spaces: a camera-side right-multiply on a c2w pose relabels the *camera*
axes, and a global left-multiply on world points relabels the *world* axes. Direction
(``Twc`` <-> ``Tcw``) conversion is matrix inverse. Scale/units stay owned by
``pipeline/stages/normalize.py`` and are never touched here.

Every conversion routes through the canonical internal representation (c2w, OpenCV
axes) so each hop is one validated primitive; validation is always on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from eval3r.core.errors import PoseConventionError
from eval3r.core.types import WorldAxes

Axes = Literal["opencv", "opengl"]
Direction = Literal["cam_to_world", "world_to_cam"]


@dataclass(frozen=True)
class PoseConvention:
    """A camera-pose convention: camera-axis handedness plus pose direction.

    ``axes`` is the camera-frame axis convention (``opencv`` = ``+X`` right / ``+Y``
    down / ``+Z`` forward; ``opengl`` = ``+X`` right / ``+Y`` up / ``+Z`` backward).
    ``direction`` is which way the stored matrix maps: ``cam_to_world`` (``Twc``, the
    camera pose in world coordinates) or ``world_to_cam`` (``Tcw``, the extrinsic).
    """

    axes: Axes
    direction: Direction


#: eval3r's internal camera-pose convention: camera-to-world, OpenCV axes.
INTERNAL_POSE_CONVENTION = PoseConvention(axes="opencv", direction="cam_to_world")

#: eval3r's internal world-frame (geometry) convention.
INTERNAL_WORLD_AXES: WorldAxes = "opencv"

#: OpenCV<->OpenGL axis flip: proper 180 degree rotation about X (involutive).
F = np.diag([1.0, -1.0, -1.0, 1.0])

# Truthfulness tolerances. atol governs absolute residuals (homogeneous row,
# round-trip); rtol scales the determinant/trace checks that are already O(1).
ATOL = 1e-8
RTOL = 1e-6
# Rotation-block acceptance gate (RᵀR and det). Real exported poses carry text/float32
# precision error: ScanNet SensReader text poses deviate by ~1e-6, float32 storage by
# up to ~1e-5. Genuine corruption (reflections, garbage) deviates by >= 1e-2, so 1e-4
# accepts legitimate real-world pose files while still rejecting invalid rotations.
# Poses are used as-is downstream; this gate only controls acceptance (task 028).
ORTHO_TOL = 1e-4
#: Above this many poses, round-trip validation is checked on a fixed-seed subsample.
_ROUND_TRIP_SUBSAMPLE = 512


# --- input validation ----------------------------------------------------------


def _as_matrices(matrices: np.ndarray, *, src: str) -> tuple[np.ndarray, bool]:
    """Return ``(N, 4, 4)`` float64 poses and whether the input was a single matrix."""
    arr = np.asarray(matrices, dtype=np.float64)
    if arr.ndim == 2:
        single = True
        arr = arr[None, ...]
    elif arr.ndim == 3:
        single = False
    else:
        raise PoseConventionError(
            f"pose input for conversion from '{src}' must be a (4, 4) matrix or an "
            f"(N, 4, 4) stack; got array with shape {arr.shape}."
        )
    if arr.shape[1:] != (4, 4):
        raise PoseConventionError(
            f"pose input for conversion from '{src}' must have trailing shape (4, 4); "
            f"got {arr.shape}."
        )
    return arr, single


def assert_finite(matrices: np.ndarray, *, context: str) -> None:
    finite = np.isfinite(matrices).all(axis=(1, 2))
    if not finite.all():
        bad = int(np.argmin(finite))
        raise PoseConventionError(
            f"{context}: pose {bad} contains NaN or Inf; convention transforms need "
            f"finite matrices (mask or drop invalid poses upstream)."
        )


def assert_homogeneous(matrices: np.ndarray, *, context: str) -> None:
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    for i, m in enumerate(matrices):
        residual = float(np.abs(m[3, :] - expected).max())
        if residual > ATOL:
            raise PoseConventionError(
                f"{context}: pose {i} bottom row is {m[3, :].tolist()}, expected "
                f"[0, 0, 0, 1] (residual {residual:.3e} > atol {ATOL:.1e}); this is not "
                f"a homogeneous 4x4 transform."
            )


def assert_orthonormal(matrices: np.ndarray, *, context: str) -> None:
    """Rotation blocks must be orthonormal with ``det = +1`` (no reflection)."""
    for i, m in enumerate(matrices):
        r = m[:3, :3]
        ortho = float(np.abs(r.T @ r - np.eye(3)).max())
        if ortho > ORTHO_TOL:
            raise PoseConventionError(
                f"{context}: pose {i} rotation block is not orthonormal "
                f"(||RᵀR - I||_max = {ortho:.3e} > {ORTHO_TOL:.1e}); a valid camera pose "
                f"has an orthonormal rotation."
            )
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > ORTHO_TOL:
            raise PoseConventionError(
                f"{context}: pose {i} rotation determinant is {det:.6f}, expected +1 "
                f"(a determinant near -1 is a reflection, not a rotation — the "
                f"conversion introduced an improper transform)."
            )


# --- the transformer -----------------------------------------------------------


class PoseConventionTransform:
    """Stateless converter between all camera-pose conventions, with validation.

    Every conversion routes through the canonical internal form (c2w, OpenCV axes):
    :meth:`to_internal` brings a source pose to canonical, :meth:`from_internal` takes
    canonical to a destination, and :meth:`convert` chains them. Each public method
    validates its inputs and its outputs and raises :class:`PoseConventionError` with
    the offending pose index, the invariant, and the numeric residual. All math is in
    float64 and vectorized over ``(N, 4, 4)`` stacks; a single ``(4, 4)`` input returns
    a single ``(4, 4)`` output.
    """

    def to_internal(self, matrices: np.ndarray, src: PoseConvention) -> np.ndarray:
        """Convert ``src``-convention poses to canonical c2w OpenCV poses."""
        arr, single = _as_matrices(matrices, src=_name(src))
        ctx = f"input ({_name(src)})"
        assert_finite(arr, context=ctx)
        assert_homogeneous(arr, context=ctx)
        assert_orthonormal(arr, context=ctx)

        out = arr
        # Direction first: bring to camera-to-world by inverting an extrinsic.
        if src.direction == "world_to_cam":
            out = _invert(out, context=f"invert w2c ({_name(src)})")
        # Axes second: right-multiply a c2w pose by F to relabel the camera axes.
        if src.axes == "opengl":
            out = out @ F

        ctx_out = f"to_internal({_name(src)})"
        assert_orthonormal(out, context=ctx_out)
        assert_homogeneous(out, context=ctx_out)
        return out[0] if single else out

    def from_internal(self, matrices: np.ndarray, dst: PoseConvention) -> np.ndarray:
        """Convert canonical c2w OpenCV poses to ``dst``-convention poses."""
        arr, single = _as_matrices(matrices, src=_name(INTERNAL_POSE_CONVENTION))
        out = arr
        # Axes first (inverse of to_internal's order): F is its own inverse.
        if dst.axes == "opengl":
            out = out @ F
        # Direction second: invert back to an extrinsic if the target is world_to_cam.
        if dst.direction == "world_to_cam":
            out = _invert(out, context=f"invert to w2c ({_name(dst)})")

        ctx_out = f"from_internal({_name(dst)})"
        assert_orthonormal(out, context=ctx_out)
        assert_homogeneous(out, context=ctx_out)
        return out[0] if single else out

    def convert(
        self, matrices: np.ndarray, src: PoseConvention, dst: PoseConvention
    ) -> np.ndarray:
        """Convert poses from ``src`` to ``dst`` convention (validated end to end)."""
        if src == dst:
            arr, single = _as_matrices(matrices, src=_name(src))
            ctx = f"passthrough ({_name(src)})"
            assert_finite(arr, context=ctx)
            assert_homogeneous(arr, context=ctx)
            assert_orthonormal(arr, context=ctx)
            return arr[0] if single else arr

        internal = self.to_internal(matrices, src)
        out = self.from_internal(internal, dst)

        self._assert_center_preserved(matrices, out, src, dst)
        self._assert_inverse_consistent(matrices, out, src, dst)
        self._assert_round_trip(matrices, out, src, dst)
        return out

    # -- truthfulness guards on a completed conversion --------------------------

    def _assert_center_preserved(
        self,
        src_in: np.ndarray,
        dst_out: np.ndarray,
        src: PoseConvention,
        dst: PoseConvention,
    ) -> None:
        """Axis-only change on same-direction c2w poses must preserve camera centre.

        Right-multiplying a c2w pose by ``F`` rotates the camera axes but leaves the
        translation column (the camera centre in world coordinates) untouched. This is
        the strongest guard against a left/right-multiply mistake. It only holds when
        neither the direction nor a world-to-cam inversion is involved.
        """
        if src.direction != "cam_to_world" or dst.direction != "cam_to_world":
            return
        a, _ = _as_matrices(src_in, src=_name(src))
        b, _ = _as_matrices(dst_out, src=_name(dst))
        residual = float(np.abs(a[:, :3, 3] - b[:, :3, 3]).max())
        if residual > ATOL:
            i = int(np.argmax(np.abs(a[:, :3, 3] - b[:, :3, 3]).max(axis=1)))
            raise PoseConventionError(
                f"convert({_name(src)} -> {_name(dst)}): camera centre of pose {i} "
                f"moved by {residual:.3e} (> atol {ATOL:.1e}) under an axis-only "
                f"change; the axis flip must not translate the camera (likely a "
                f"left/right-multiply mistake)."
            )

    def _assert_inverse_consistent(
        self,
        src_in: np.ndarray,
        dst_out: np.ndarray,
        src: PoseConvention,
        dst: PoseConvention,
    ) -> None:
        """Source and destination poses must describe the same physical camera.

        For a direction change (``Twc`` <-> ``Tcw``) the strongest guard is that both
        sides collapse to the *same* canonical c2w pose: the source and the output
        differ only in how they encode one camera, never in the camera itself.
        """
        if src.direction == dst.direction:
            return
        src_canonical = self.to_internal(src_in, src)
        dst_canonical = self.to_internal(dst_out, dst)
        a, _ = _as_matrices(src_canonical, src=_name(INTERNAL_POSE_CONVENTION))
        b, _ = _as_matrices(dst_canonical, src=_name(INTERNAL_POSE_CONVENTION))
        residual = float(np.abs(a - b).max())
        if residual > 1e-6:
            i = int(np.argmax(np.abs(a - b).reshape(a.shape[0], -1).max(axis=1)))
            raise PoseConventionError(
                f"convert({_name(src)} -> {_name(dst)}): a direction change made pose "
                f"{i} describe a different camera (canonical c2w residual {residual:.3e} "
                f"> 1e-6); the inverse is inconsistent."
            )

    def _assert_round_trip(
        self,
        src_in: np.ndarray,
        dst_out: np.ndarray,
        src: PoseConvention,
        dst: PoseConvention,
    ) -> None:
        """``convert(convert(M, src, dst), dst, src)`` must recover ``M``."""
        arr, _ = _as_matrices(src_in, src=_name(src))
        n = arr.shape[0]
        if n > _ROUND_TRIP_SUBSAMPLE:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=_ROUND_TRIP_SUBSAMPLE, replace=False)
        else:
            idx = np.arange(n)
        back = self.from_internal(self.to_internal(dst_out, dst), src)
        back_arr, _ = _as_matrices(back, src=_name(src))
        residual = float(np.abs(arr[idx] - back_arr[idx]).max())
        if residual > 1e-6:
            raise PoseConventionError(
                f"convert({_name(src)} -> {_name(dst)}): round-trip back to the source "
                f"convention differs from the input by {residual:.3e} (> 1e-6); the "
                f"conversion is not invertible as expected."
            )

    # -- TUM trajectory path ----------------------------------------------------

    def tum_rows_to_matrices(self, rows: np.ndarray) -> np.ndarray:
        """Convert ``(N, 8)`` TUM rows ``[t x y z qx qy qz qw]`` to ``(N, 4, 4)`` poses.

        Quaternions are scalar-last (``[qx, qy, qz, qw]``), matching evo and
        ``scipy.spatial.transform.Rotation``. The timestamp column is dropped here and
        preserved separately by :meth:`convert_tum_rows`.
        """
        arr = _as_tum_rows(rows)
        quats = arr[:, 4:8]
        norms = np.linalg.norm(quats, axis=1)
        if not np.all(norms > 1e-8):
            bad = int(np.argmin(norms))
            raise PoseConventionError(
                f"TUM row {bad} has a near-zero quaternion (norm {norms[bad]:.3e}); "
                f"cannot build a rotation."
            )
        rot = Rotation.from_quat(quats)  # scipy normalizes internally
        mats = np.repeat(np.eye(4)[None, ...], arr.shape[0], axis=0)
        mats[:, :3, :3] = rot.as_matrix()
        mats[:, :3, 3] = arr[:, 1:4]
        return mats

    def matrices_to_tum_rows(
        self, timestamps: np.ndarray, matrices: np.ndarray
    ) -> np.ndarray:
        """Convert ``(N, 4, 4)`` poses + timestamps back to ``(N, 8)`` TUM rows."""
        arr, _ = _as_matrices(matrices, src=_name(INTERNAL_POSE_CONVENTION))
        assert_orthonormal(arr, context="matrices_to_tum_rows")
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.shape[0] != arr.shape[0]:
            raise PoseConventionError(
                f"timestamp count ({ts.shape[0]}) does not match pose count "
                f"({arr.shape[0]}) when writing TUM rows."
            )
        quats = Rotation.from_matrix(arr[:, :3, :3]).as_quat()  # scalar-last, unit
        norms = np.linalg.norm(quats, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            bad = int(np.argmax(np.abs(norms - 1.0)))
            raise PoseConventionError(
                f"produced a non-unit quaternion at row {bad} (norm {norms[bad]:.6f}); "
                f"rotation-to-quaternion conversion is inconsistent."
            )
        rows = np.empty((arr.shape[0], 8), dtype=np.float64)
        rows[:, 0] = ts
        rows[:, 1:4] = arr[:, :3, 3]
        rows[:, 4:8] = quats
        return rows

    def convert_tum_rows(
        self, rows: np.ndarray, src: PoseConvention, dst: PoseConvention
    ) -> np.ndarray:
        """Convert TUM ``(N, 8)`` rows from ``src`` to ``dst``, preserving timestamps."""
        arr = _as_tum_rows(rows)
        timestamps = arr[:, 0].copy()
        mats = self.tum_rows_to_matrices(arr)
        converted = self.convert(mats, src, dst)
        conv_arr, _ = _as_matrices(converted, src=_name(dst))
        return self.matrices_to_tum_rows(timestamps, conv_arr)

    # -- world-frame (geometry) path --------------------------------------------

    def world_transform(self, src_world: WorldAxes, dst_world: WorldAxes) -> np.ndarray:
        """Return the ``(4, 4)`` global transform mapping ``src_world`` world axes to ``dst_world``.

        Identity when the frames match, else the involution ``F`` applied as a global
        left-multiply on world points (``LoadedGeometry.transformed`` uses ``R`` and
        ``t`` blocks directly). ``opencv`` <-> ``opengl`` is a proper rotation, so
        geometry is rotated, never reflected.
        """
        if src_world not in ("opencv", "opengl") or dst_world not in ("opencv", "opengl"):
            raise PoseConventionError(
                f"world-frame conversion supports only 'opencv' and 'opengl'; got "
                f"src='{src_world}', dst='{dst_world}'."
            )
        if src_world == dst_world:
            return np.eye(4)
        return F.copy()


# --- helpers -------------------------------------------------------------------


def _name(convention: PoseConvention) -> str:
    return f"{convention.axes}/{convention.direction}"


def _invert(matrices: np.ndarray, *, context: str) -> np.ndarray:
    """Invert each ``(4, 4)`` rigid pose analytically (Rᵀ, -Rᵀt), then verify."""
    r = matrices[:, :3, :3]
    t = matrices[:, :3, 3]
    rt = np.transpose(r, (0, 2, 1))
    out = np.repeat(np.eye(4)[None, ...], matrices.shape[0], axis=0)
    out[:, :3, :3] = rt
    out[:, :3, 3] = -np.einsum("nij,nj->ni", rt, t)
    prod = matrices @ out
    residual = float(np.abs(prod - np.eye(4)).max())
    if residual > 1e-6:
        raise PoseConventionError(
            f"{context}: analytic inverse deviates from identity by {residual:.3e} "
            f"(> 1e-6); the input pose is likely not a rigid transform."
        )
    return out


def read_tum_rows(path: Path) -> np.ndarray:
    """Read a TUM trajectory text file into an ``(N, 8)`` float64 array.

    Rows are ``timestamp x y z qx qy qz qw`` (whitespace-separated). Blank lines and
    ``#`` comments are skipped. Raises :class:`PoseConventionError` naming the file when
    it is missing, empty, or not 8 columns — the convention step must never guess.
    """
    path = Path(path)
    if not path.is_file():
        raise PoseConventionError(
            f"trajectory file for convention conversion does not exist: {path}."
        )
    try:
        raw = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError as exc:
        raise PoseConventionError(
            f"trajectory file {path} is not parseable as TUM text "
            f"(timestamp x y z qx qy qz qw): {exc}."
        ) from exc
    if raw.size == 0 or raw.ndim != 2 or raw.shape[1] != 8:
        raise PoseConventionError(
            f"trajectory file {path} must have 8 columns "
            f"(timestamp x y z qx qy qz qw); parsed shape {raw.shape}."
        )
    return np.asarray(raw, dtype=np.float64)


def _as_tum_rows(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 8 or arr.shape[0] == 0:
        raise PoseConventionError(
            f"a TUM trajectory must be a non-empty (N, 8) array of rows "
            f"[timestamp x y z qx qy qz qw]; got shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise PoseConventionError(
            "a TUM trajectory array must be finite; it contains NaN or Inf values."
        )
    return arr
