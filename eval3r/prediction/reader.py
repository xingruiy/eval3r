"""PredictionReader — load + verify a prediction directory written by PredictionWriter."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

from eval3r.io.geometry import MeshData, PointCloudData, load_mesh, load_point_cloud
from eval3r.io.trajectory import Trajectory, load_trajectory_tum
from eval3r.prediction._hash import sha256_file
from eval3r.prediction.manifest import MANIFEST_FILENAME, Artifact, Manifest
from eval3r.utils.errors import (
    CorruptedArtifactError,
    ManifestError,
    MissingArtifactError,
)
from eval3r.utils.typing import PathLike


class PredictionReader:
    """Lazy reader for a prediction directory; verifies sha256 on artifact access."""

    def __init__(self, root: PathLike, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        self._verify = verify_hashes
        manifest_path = self.root / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise MissingArtifactError(
                f"No {MANIFEST_FILENAME} found in {self.root}. "
                f"Pass the directory containing the manifest, not its parent."
            )
        try:
            data = json.loads(manifest_path.read_text())
            self.manifest: Manifest = Manifest.model_validate(data)
        except Exception as e:
            raise ManifestError(f"Failed to parse manifest at {manifest_path}: {e}") from e

    # -- artifact access --------------------------------------------------
    @cached_property
    def points(self) -> PointCloudData:
        art = self._require(self.manifest.geometry.point_cloud, "geometry.point_cloud")
        return load_point_cloud(self._resolve(art))

    @cached_property
    def mesh(self) -> MeshData:
        art = self._require(self.manifest.geometry.mesh, "geometry.mesh")
        return load_mesh(self._resolve(art))

    @cached_property
    def poses(self) -> Trajectory:
        art = self._require(self.manifest.trajectory.tum, "trajectory.tum")
        return load_trajectory_tum(self._resolve(art), convention=self.manifest.pose_convention.value)

    @cached_property
    def intrinsics(self) -> dict[str, Any]:
        art = self._require(self.manifest.cameras.intrinsics, "cameras.intrinsics")
        path = self._resolve(art)
        return json.loads(path.read_text())

    # -- helpers ----------------------------------------------------------
    def has(self, section: str) -> bool:
        try:
            self._require(self._artifact_for(section), section)
            return True
        except MissingArtifactError:
            return False

    def _artifact_for(self, section: str) -> Artifact | None:
        m = self.manifest
        mapping: dict[str, Artifact | None] = {
            "geometry.mesh": m.geometry.mesh,
            "geometry.point_cloud": m.geometry.point_cloud,
            "trajectory.tum": m.trajectory.tum,
            "trajectory.kitti": m.trajectory.kitti,
            "cameras.intrinsics": m.cameras.intrinsics,
            "cameras.poses": m.cameras.poses,
        }
        if section not in mapping:
            raise KeyError(f"Unknown manifest section: {section}")
        return mapping[section]

    def _require(self, art: Artifact | None, name: str) -> Artifact:
        if art is None:
            raise MissingArtifactError(
                f"Prediction at {self.root} has no '{name}' section in its manifest."
            )
        return art

    def _resolve(self, art: Artifact) -> Path:
        path = self.root / art.path
        if not path.exists():
            raise MissingArtifactError(
                f"Manifest references {art.path} but file is missing under {self.root}."
            )
        if self._verify:
            actual = sha256_file(path)
            if actual != art.sha256:
                raise CorruptedArtifactError(
                    f"sha256 mismatch for {art.path}: "
                    f"manifest={art.sha256[:12]}…, actual={actual[:12]}…"
                )
        return path
