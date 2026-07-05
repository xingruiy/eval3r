"""pycolmap camera backend (``camera`` kind).

Loads COLMAP models (text or binary directories with ``cameras.txt`` /
``images.txt`` / ``points3D.txt``) through **pycolmap**, so every camera model
COLMAP defines is parsed by the reference implementation rather than a hand-rolled
parser (``.agent/backends.md`` camera rules; ETH3D uses COLMAP text format).

Convention normalization: COLMAP stores world-to-camera poses with OpenCV-style
camera axes (+X right, +Y down, +Z forward). Poses are inverted here to eval3r's
internal camera-to-world convention; the source format (``world_to_cam_colmap``)
is recorded on the returned camera set, never discarded.

Pinhole honesty: :meth:`PycolmapCameraBackend.pinhole_intrinsics` returns a 3x3 K
only for actual pinhole models (``PINHOLE`` / ``SIMPLE_PINHOLE``). Any other model
raises an explicit error naming the camera and model instead of silently
approximating it as pinhole (CLAUDE.md ETH3D caution). Callers that only need the
poses and model names can use :func:`load_cameras` freely — non-pinhole models are
fully parsed, they are just never *reduced* to pinhole silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval3r.core.errors import DatasetError
from eval3r.core.registry import BackendInfo

# COLMAP models that are exactly representable by a 3x3 pinhole K.
PINHOLE_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})


@dataclass(frozen=True)
class ColmapCamera:
    """One COLMAP camera: model name, image size, and native parameter vector."""

    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def is_pinhole(self) -> bool:
        return self.model in PINHOLE_MODELS


@dataclass(frozen=True)
class ColmapImagePose:
    """One registered image: name, camera link, and normalized cam-to-world pose."""

    image_id: int
    name: str
    camera_id: int
    cam_to_world: np.ndarray  # 4x4, OpenCV axes, metres (COLMAP world scale)


@dataclass(frozen=True)
class ColmapCameraSet:
    """All cameras and image poses of one COLMAP model, convention-normalized."""

    path: Path
    cameras: dict[int, ColmapCamera]
    images: list[ColmapImagePose]
    source_pose_format: str = "world_to_cam_colmap"
    normalized_convention: str = "cam_to_world_opencv_meters"
    notes: list[str] = field(default_factory=list)

    @property
    def camera_models(self) -> dict[int, str]:
        return {cid: cam.model for cid, cam in self.cameras.items()}

    @property
    def non_pinhole_models(self) -> dict[int, str]:
        return {cid: cam.model for cid, cam in self.cameras.items() if not cam.is_pinhole}


class PycolmapCameraBackend:
    """Camera backend delegating COLMAP model parsing to pycolmap."""

    name = "pycolmap"

    def backend_info(self) -> BackendInfo:
        import pycolmap

        return BackendInfo(
            kind="camera",
            name=self.name,
            library="pycolmap",
            version=getattr(pycolmap, "__version__", "unknown"),
            approximate=False,
        )

    def load_cameras(self, path: Path) -> ColmapCameraSet:
        """Load a COLMAP model directory and normalize poses to cam-to-world OpenCV.

        ``path`` must contain ``cameras.txt`` / ``images.txt`` / ``points3D.txt``
        (or their ``.bin`` equivalents). Failures name the path and the underlying
        pycolmap error.
        """
        import pycolmap

        path = Path(path)
        if not path.is_dir():
            raise DatasetError(
                f"COLMAP model directory does not exist: {path}. Expected a directory with "
                f"cameras.txt / images.txt / points3D.txt (or .bin)."
            )
        try:
            reconstruction = pycolmap.Reconstruction(path)
        except Exception as exc:
            raise DatasetError(
                f"pycolmap could not read the COLMAP model at {path}: {exc}"
            ) from exc

        cameras: dict[int, ColmapCamera] = {}
        for camera_id, camera in reconstruction.cameras.items():
            cameras[int(camera_id)] = ColmapCamera(
                camera_id=int(camera_id),
                model=camera.model.name,
                width=int(camera.width),
                height=int(camera.height),
                params=np.asarray(camera.params, dtype=np.float64),
            )

        images: list[ColmapImagePose] = []
        for image_id, image in reconstruction.images.items():
            cam_from_world = image.cam_from_world
            if callable(cam_from_world):  # pycolmap >= 3.12 exposes it as a method
                cam_from_world = cam_from_world()
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
            c2w = np.linalg.inv(w2c)
            images.append(
                ColmapImagePose(
                    image_id=int(image_id),
                    name=image.name,
                    camera_id=int(image.camera_id),
                    cam_to_world=c2w,
                )
            )
        images.sort(key=lambda im: im.image_id)

        camera_set = ColmapCameraSet(path=path, cameras=cameras, images=images)
        non_pinhole = camera_set.non_pinhole_models
        if non_pinhole:
            models = ", ".join(f"camera {cid}: {m}" for cid, m in sorted(non_pinhole.items()))
            camera_set.notes.append(
                f"non-pinhole camera model(s) present ({models}); poses are exact, but these "
                f"cameras cannot be reduced to a pinhole K and any pinhole-only consumer must "
                f"fail explicitly rather than approximate."
            )
        return camera_set

    def pinhole_intrinsics(self, camera: ColmapCamera) -> np.ndarray:
        """3x3 K for an actual pinhole camera; explicit error for any other model."""
        if not camera.is_pinhole:
            raise DatasetError(
                f"camera {camera.camera_id} uses the COLMAP model '{camera.model}', which is "
                f"not a simple pinhole model ({', '.join(sorted(PINHOLE_MODELS))}). Refusing to "
                f"approximate it as pinhole; use a consumer that supports this model via "
                f"pycolmap, or exclude the camera by protocol."
            )
        if camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        else:
            fx, fy, cx, cy = camera.params
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
