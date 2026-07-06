"""Canonical filename and consistency rules for the eval3r-native prediction layout.

The layout maps each :class:`~eval3r.core.manifest.ScenePredictionEntry` path field
onto a canonical name inside ``<pred_root>/<scene_id>/``. Copied files keep their
source suffix (a mesh may be ``mesh.ply`` or ``mesh.obj``); array inputs written by
eval3r always get the canonical extension. Manifest paths stay relative to the
prediction root so the directory is relocatable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from eval3r.core.errors import PredictionLayoutError
from eval3r.core.types import PredictionModality
from eval3r.datasets.base import file_fingerprint

LAYOUT_NAME = "eval3r-native-v1"

# ScenePredictionEntry field -> canonical filename stem (single-file fields).
FILE_FIELD_STEMS: dict[str, str] = {
    "mesh": "mesh",
    "pointcloud": "pointcloud",
    "pointmap": "pointmap",
    "depth": "depth",
    "trajectory": "trajectory_tum",
    "camera_file": "cameras",
    "confidence": "confidence",
}

# ScenePredictionEntry field -> canonical directory name (directory fields).
DIR_FIELD_NAMES: dict[str, str] = {
    "depth_dir": "depth",
    "confidence_dir": "confidence",
}

ENTRY_PATH_FIELDS: tuple[str, ...] = (*FILE_FIELD_STEMS, *DIR_FIELD_NAMES)

# Canonical extension when eval3r writes the file from an array. Fields absent
# here are copy-only: eval3r owns no writer for them (meshes are never built by
# eval3r; depth and camera files are produced by the method's own tooling).
ARRAY_FIELD_EXTENSIONS: dict[str, str] = {
    "pointcloud": ".ply",
    "pointmap": ".npy",
    "trajectory": ".txt",
    "confidence": ".npy",
}

# Declared prediction modality -> the entry field every scene must provide.
# All other fields are auxiliary and always allowed.
MODALITY_REQUIRED_FIELD: dict[str, str] = {
    "mesh": "mesh",
    "pointcloud": "pointcloud",
    "pointmap": "pointmap",
    "single_depth": "depth",
    "depth_sequence": "depth_dir",
    "camera_trajectory": "trajectory",
}


def required_field(modality: PredictionModality) -> str:
    """The entry field the declared modality requires in every scene.

    Raises for modalities the eval3r-native layout does not define (currently
    ``colmap_reconstruction``: a COLMAP model is a directory of dataset-specific
    files with no single canonical prediction file).
    """
    field = MODALITY_REQUIRED_FIELD.get(modality)
    if field is None:
        raise PredictionLayoutError(
            f"prediction modality '{modality}' has no eval3r-native layout: the layout "
            f"defines canonical files for {sorted(MODALITY_REQUIRED_FIELD)}. Write such "
            f"predictions with a hand-authored manifest instead of PredictionWriter."
        )
    return field


def canonical_relpath(scene_id: str, field: str, suffix: str) -> Path:
    """Manifest-relative path for one entry field of one scene.

    ``suffix`` is the file extension to keep (source suffix for copies, canonical
    extension for array inputs); it is ignored for directory fields.
    """
    if field in DIR_FIELD_NAMES:
        return Path(scene_id) / DIR_FIELD_NAMES[field]
    if field in FILE_FIELD_STEMS:
        return Path(scene_id) / f"{FILE_FIELD_STEMS[field]}{suffix}"
    raise PredictionLayoutError(
        f"'{field}' is not a prediction entry path field; expected one of "
        f"{sorted(ENTRY_PATH_FIELDS)}."
    )


def directory_fingerprint(path: Path) -> str | None:
    """Joint content hash of every file under a directory, or ``None`` if absent.

    Hashes ``<relative posix path>:<sha256 hex>`` lines for the sorted file list, so
    renaming, adding, removing, or editing any frame changes the fingerprint. Used
    for ``depth_dir`` / ``confidence_dir`` entries.
    """
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for member in sorted(p for p in path.rglob("*") if p.is_file()):
        member_hash = file_fingerprint(member)
        digest.update(f"{member.relative_to(path).as_posix()}:{member_hash}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def field_fingerprint(field: str, path: Path) -> str | None:
    """Fingerprint one resolved entry field: file hash, or joint hash for dir fields."""
    if field in DIR_FIELD_NAMES:
        return directory_fingerprint(path)
    return file_fingerprint(path)
