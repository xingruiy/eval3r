"""Read and verify an eval3r-native prediction directory.

:func:`check_prediction_dir` resolves every declared per-scene path against the
prediction root and collects every problem (missing files, fingerprint
mismatches) instead of stopping at the first one — the CLI renders them as a
per-scene table. :func:`read_prediction_dir` is the strict form: it raises a
single :class:`PredictionLayoutError` listing everything that is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from eval3r.core.errors import PredictionLayoutError
from eval3r.core.manifest import PredictionManifest
from eval3r.predictions.layout import (
    DIR_FIELD_NAMES,
    ENTRY_PATH_FIELDS,
    field_fingerprint,
)


@dataclass
class PredictionCheck:
    """Resolution outcome for a prediction directory.

    ``scenes`` maps scene id -> entry field -> absolute path for every declared
    path that resolved; ``issues`` maps scene id -> list of human-readable
    problems (missing file, fingerprint mismatch, missing fingerprint record).
    A directory is valid iff ``issues`` is empty.
    """

    root: Path
    manifest: PredictionManifest
    scenes: dict[str, dict[str, Path]] = field(default_factory=dict)
    issues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def load_prediction_manifest(root: Path | str) -> PredictionManifest:
    """Load and schema-validate ``<root>/manifest.yaml`` with explicit errors."""
    root = Path(root)
    manifest_path = root / "manifest.yaml"
    if not root.is_dir():
        raise PredictionLayoutError(
            f"prediction root {root} is not a directory; expected an eval3r-native "
            f"prediction directory containing manifest.yaml."
        )
    if not manifest_path.is_file():
        raise PredictionLayoutError(
            f"prediction root {root} has no manifest.yaml; an eval3r-native "
            f"prediction directory always carries its manifest at the root "
            f"(write it with PredictionWriter, or author one by hand)."
        )
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PredictionLayoutError(
            f"manifest {manifest_path} is not valid YAML:\n{exc}"
        ) from exc
    try:
        return PredictionManifest.model_validate(data)
    except ValidationError as exc:
        raise PredictionLayoutError(
            f"manifest {manifest_path} failed prediction-manifest schema "
            f"validation:\n{exc}"
        ) from exc


def check_prediction_dir(root: Path | str, *, verify: bool = False) -> PredictionCheck:
    """Resolve every scene of a prediction directory, collecting all problems.

    With ``verify=True``, every resolved file is re-hashed against the
    fingerprint recorded in the scene entry's ``metadata.fingerprints``; a
    missing fingerprint record is itself reported (verification cannot pass
    vacuously). An unreadable or schema-invalid manifest still raises, because
    nothing can be checked without one.
    """
    root = Path(root)
    manifest = load_prediction_manifest(root)
    check = PredictionCheck(root=root, manifest=manifest)

    for scene_id, entry in manifest.scenes.items():
        problems: list[str] = []
        resolved: dict[str, Path] = {}
        recorded = entry.metadata.get("fingerprints", {})
        if not isinstance(recorded, dict):
            problems.append(
                f"metadata.fingerprints is {type(recorded).__name__}, expected a "
                f"mapping of entry field to 'sha256:...' digest"
            )
            recorded = {}
        for field_name in ENTRY_PATH_FIELDS:
            declared = getattr(entry, field_name)
            if declared is None:
                continue
            path = declared if declared.is_absolute() else root / declared
            is_dir_field = field_name in DIR_FIELD_NAMES
            exists = path.is_dir() if is_dir_field else path.is_file()
            if not exists:
                kind = "directory" if is_dir_field else "file"
                problems.append(f"field '{field_name}': missing {kind} {path}")
                continue
            resolved[field_name] = path
            if not verify:
                continue
            expected = recorded.get(field_name)
            if expected is None:
                problems.append(
                    f"field '{field_name}': no fingerprint recorded in "
                    f"metadata.fingerprints, cannot verify {path}"
                )
                continue
            actual = field_fingerprint(field_name, path)
            if actual != expected:
                problems.append(
                    f"field '{field_name}': fingerprint mismatch for {path} — "
                    f"recorded {expected}, actual {actual}; the file changed after "
                    f"the manifest was written"
                )
        check.scenes[scene_id] = resolved
        if problems:
            check.issues[scene_id] = problems
    return check


def read_prediction_dir(root: Path | str, *, verify: bool = False) -> PredictionCheck:
    """Strictly resolve a prediction directory to absolute per-scene paths.

    Raises one :class:`PredictionLayoutError` naming every scene, field, and
    path that failed (never just the first problem). ``verify=True``
    additionally re-hashes every file against the recorded fingerprints.
    """
    check = check_prediction_dir(root, verify=verify)
    if check.issues:
        lines = [
            f"prediction directory {check.root} failed "
            f"{'verification' if verify else 'resolution'} for "
            f"{len(check.issues)}/{len(check.manifest.scenes)} scenes:"
        ]
        for scene_id, problems in check.issues.items():
            for problem in problems:
                lines.append(f"  scene '{scene_id}': {problem}")
        raise PredictionLayoutError("\n".join(lines))
    return check
