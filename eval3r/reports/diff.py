"""Run result diffing (``diff_runs`` / ``e3r diff``).

Compares two run directories' ``results.json`` files. Strict mode (the default)
refuses runs with different protocol hashes — metric numbers produced under
different protocols are not comparable, so a strict diff between them would be
misleading. A loose comparison can be requested explicitly and is labeled
non-strict in the returned :class:`RunDiff` and in every rendering of it.

Beyond the hash, the diff warns on every comparability trigger from
``.agent/reproducibility.md`` ("Report comparability"): GT provenance, local
evaluation status, scene coverage, failure policy, alignment mode, confidence
policy, sampling counts, and backend officialness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval3r.core.errors import RunComparisonError
from eval3r.core.result import RunResult
from eval3r.core.schema import E3RModel
from eval3r.reports.json import read_run_result_json


class RunIdentity(E3RModel):
    """The identity and coverage facts of one run, as shown in diff output."""

    run_dir: str
    method: str | None
    dataset: str
    variant: str | None
    split: str | None
    protocol: str
    protocol_version: str
    protocol_hash: str
    fidelity: str
    failure_policy: str
    n_scenes_expected: int
    n_scenes_evaluated: int
    n_scenes_failed: int


class MetricDelta(E3RModel):
    """One metric compared across the two runs (``delta = value_b - value_a``)."""

    name: str
    value_a: float | None
    value_b: float | None
    delta: float | None


class DiffWarning(E3RModel):
    """One comparability warning: the field that differs and both values."""

    field: str
    reason: str
    value_a: str
    value_b: str


class RunDiff(E3RModel):
    """Structured comparison of two runs; ``strict`` records the comparison mode."""

    strict: bool
    run_a: RunIdentity
    run_b: RunIdentity
    warnings: list[DiffWarning] = []
    metrics: list[MetricDelta] = []
    per_scene: dict[str, list[MetricDelta]] = {}
    scenes_only_in_a: list[str] = []
    scenes_only_in_b: list[str] = []


def load_run_result(run_dir: str | Path) -> RunResult:
    """Load a run directory's ``results.json`` (or a direct path to one)."""
    path = Path(run_dir)
    results = path if path.is_file() else path / "results.json"
    if not results.is_file():
        raise RunComparisonError(
            f"run '{path}' has no results.json to compare "
            f"(looked for: {results}). Pass a run directory written by eval3r, "
            f"or the results.json file itself."
        )
    try:
        return read_run_result_json(results)
    except ValueError as exc:
        raise RunComparisonError(
            f"results.json in run '{path}' failed RunResult validation: {exc}"
        ) from exc


def _identity(run_dir: str | Path, result: RunResult) -> RunIdentity:
    return RunIdentity(
        run_dir=str(run_dir),
        method=result.method,
        dataset=result.dataset.dataset,
        variant=result.dataset.variant,
        split=result.split,
        protocol=result.protocol,
        protocol_version=result.protocol_version,
        protocol_hash=result.protocol_hash,
        fidelity=result.fidelity,
        failure_policy=result.failure_policy.policy,
        n_scenes_expected=result.n_scenes_expected,
        n_scenes_evaluated=result.n_scenes_evaluated,
        n_scenes_failed=len(result.failed_scenes),
    )


def _official_eval_summary(result: RunResult) -> str:
    entry = result.backend_versions.get("official_eval")
    if entry is None:
        return f"fidelity={result.fidelity}, official_eval=(none)"
    if isinstance(entry, dict):
        name = entry.get("name", "?")
        version = entry.get("version", "?")
        return f"fidelity={result.fidelity}, official_eval={name} {version}"
    return f"fidelity={result.fidelity}, official_eval={entry}"


def _coverage_summary(result: RunResult) -> str:
    return (
        f"expected={result.n_scenes_expected}, "
        f"evaluated={result.n_scenes_evaluated}, "
        f"failed={len(result.failed_scenes)}"
    )


def _sampling_summary(result: RunResult) -> str:
    return (
        f"pred: {result.sampling.pred.method} n={result.sampling.pred.n_points}; "
        f"gt: {result.sampling.gt.method} n={result.sampling.gt.n_points}"
    )


def _adaptation_summary(result: RunResult) -> str:
    if result.adaptation is None:
        return "adaptation=(none)"
    return str(result.adaptation.model_dump(mode="json", exclude_none=True))


def build_warnings(a: RunResult, b: RunResult) -> list[DiffWarning]:
    """Comparability warnings per ``.agent/reproducibility.md``."""
    checks: list[tuple[str, str, str, str]] = [
        (
            "protocol_hash",
            "metric numbers under different protocols are not comparable",
            a.protocol_hash,
            b.protocol_hash,
        ),
        (
            "ground_truth_provenance",
            "the runs were scored against different kinds of ground truth",
            f"provenance={a.ground_truth.provenance}, independence={a.ground_truth.independence}",
            f"provenance={b.ground_truth.provenance}, independence={b.ground_truth.independence}",
        ),
        (
            "local_evaluation_status",
            "one run's split may not be fully locally evaluable",
            a.local_evaluation.status,
            b.local_evaluation.status,
        ),
        (
            "scene_coverage",
            "aggregates cover different scene sets; averages are not comparable",
            _coverage_summary(a),
            _coverage_summary(b),
        ),
        (
            "failure_policy",
            "failed scenes contribute differently to the aggregates",
            a.failure_policy.policy,
            b.failure_policy.policy,
        ),
        (
            "alignment_mode",
            "errors were measured after different alignment",
            a.alignment.mode,
            b.alignment.mode,
        ),
        (
            "adaptation",
            "runs share protocol identity but adapted predictions differently",
            _adaptation_summary(a),
            _adaptation_summary(b),
        ),
        (
            "confidence_policy",
            "predictions were filtered differently before scoring",
            a.confidence_policy.policy,
            b.confidence_policy.policy,
        ),
        (
            "sampling",
            "metrics were computed on differently sampled point sets",
            _sampling_summary(a),
            _sampling_summary(b),
        ),
        (
            "backend_officialness",
            "one run may carry official-fidelity numbers and the other not",
            _official_eval_summary(a),
            _official_eval_summary(b),
        ),
    ]
    return [
        DiffWarning(field=field, reason=reason, value_a=va, value_b=vb)
        for field, reason, va, vb in checks
        if va != vb
    ]


def _metric_deltas(
    values_a: dict[str, float | None], values_b: dict[str, float | None]
) -> list[MetricDelta]:
    names = list(values_a)
    names += [n for n in values_b if n not in values_a]
    deltas: list[MetricDelta] = []
    for name in names:
        va = values_a.get(name)
        vb = values_b.get(name)
        delta = (vb - va) if (va is not None and vb is not None) else None
        deltas.append(MetricDelta(name=name, value_a=va, value_b=vb, delta=delta))
    return deltas


def _per_scene_values(result: RunResult) -> dict[str, dict[str, float | None]]:
    scenes: dict[str, dict[str, float | None]] = {}
    for mr in result.per_scene_metrics:
        scene = mr.scene_id or "<unknown>"
        label = f"{scene}/{mr.frame_id}" if mr.frame_id else scene
        scenes.setdefault(label, {})[mr.name] = mr.value
    for failure in result.failed_scenes:
        scenes.setdefault(failure.scene_id, {})
    return scenes


def diff_runs(
    run_a: str | Path,
    run_b: str | Path,
    *,
    loose: bool = False,
) -> RunDiff:
    """Compare two run directories.

    Strict mode (default) raises :class:`RunComparisonError` when protocol hashes
    differ. ``loose=True`` allows the comparison but the result is labeled
    non-strict (``strict=False``) and carries the full warning list.
    """
    result_a = load_run_result(run_a)
    result_b = load_run_result(run_b)

    if result_a.protocol_hash != result_b.protocol_hash and not loose:
        raise RunComparisonError(
            f"strict diff refused: the runs have different protocol hashes, so their "
            f"metric numbers are not comparable.\n"
            f"  run A: {run_a}\n    protocol {result_a.protocol} "
            f"v{result_a.protocol_version}, hash {result_a.protocol_hash}\n"
            f"  run B: {run_b}\n    protocol {result_b.protocol} "
            f"v{result_b.protocol_version}, hash {result_b.protocol_hash}\n"
            f"Re-run one side under the other's protocol, or request an explicitly "
            f"non-strict comparison (diff_runs(..., loose=True) / e3r diff --loose)."
        )

    scenes_a = _per_scene_values(result_a)
    scenes_b = _per_scene_values(result_b)
    common = [label for label in scenes_a if label in scenes_b]
    per_scene = {
        label: _metric_deltas(scenes_a[label], scenes_b[label]) for label in common
    }

    return RunDiff(
        strict=not loose,
        run_a=_identity(run_a, result_a),
        run_b=_identity(run_b, result_b),
        warnings=build_warnings(result_a, result_b),
        metrics=_metric_deltas(result_a.metrics, result_b.metrics),
        per_scene=per_scene,
        scenes_only_in_a=sorted(set(scenes_a) - set(scenes_b)),
        scenes_only_in_b=sorted(set(scenes_b) - set(scenes_a)),
    )


def diff_as_dict(diff: RunDiff) -> dict[str, Any]:
    """JSON-serializable view of a diff (for programmatic consumers)."""
    return diff.model_dump(mode="json")
