"""Format-agnostic report tables built from a :class:`RunResult`.

Markdown/LaTeX/HTML reports (task 016) all render the same :class:`ReportData`,
built here purely from an already-computed ``RunResult`` — report renderers never
recompute metrics. Every report surfaces the protocol identity and hash, fidelity,
ground-truth provenance, local-evaluation status, the alignment / confidence /
sampling policies in effect, full scene coverage, failure reasons, and a
partial-coverage banner whenever aggregates do not cover every expected scene
(``.agent/reproducibility.md`` "Scene coverage and failures").
"""

from __future__ import annotations

from dataclasses import dataclass

from eval3r.core.result import MetricResult, RunResult
from eval3r.core.schema import SamplingSideSpec

#: Placeholder shown for a metric value that is absent because the scene failed.
MISSING_VALUE = "n/a (failed)"


@dataclass(frozen=True)
class CoverageInfo:
    """Scene coverage of a run: the numbers every report format must show."""

    expected: int
    evaluated: int
    failed: int
    failure_policy: str

    @property
    def partial(self) -> bool:
        return self.evaluated < self.expected or self.failed > 0


@dataclass(frozen=True)
class ReportData:
    """Everything a report renderer needs, already reduced to strings."""

    title: str
    header: list[tuple[str, str]]
    coverage: CoverageInfo
    banner: str | None
    aggregate_columns: list[str]
    aggregate_rows: list[list[str]]
    per_scene_columns: list[str]
    per_scene_rows: list[list[str]]
    failures: list[tuple[str, str, str]]  # (scene_id, stage, reason)


def format_value(value: float | None) -> str:
    if value is None:
        return MISSING_VALUE
    return f"{value:.6g}"


def _sampling_summary(side: SamplingSideSpec) -> str:
    if side.method in ("none", "all_points"):
        return side.method
    n = "?" if side.n_points is None else str(side.n_points)
    return f"{side.method} (n={n}, seed={side.seed})"


def coverage_banner(coverage: CoverageInfo) -> str | None:
    """Partial-coverage banner text, or ``None`` when every scene was evaluated."""
    if not coverage.partial:
        return None
    return (
        f"PARTIAL COVERAGE: {coverage.evaluated} of {coverage.expected} scenes "
        f"evaluated, {coverage.failed} failed (failure policy: "
        f"{coverage.failure_policy}). Aggregate metrics do not cover every "
        f"expected scene — see the failure table for per-scene reasons."
    )


def _backend_summary(result: RunResult) -> str:
    parts: list[str] = []
    for kind, entry in sorted(result.backend_versions.items()):
        if isinstance(entry, dict):
            name = entry.get("name", "?")
            version = entry.get("version", "?")
            parts.append(f"{kind}={name} {version}")
        else:
            parts.append(f"{kind}={entry}")
    return "; ".join(parts) if parts else "(none recorded)"


def build_header(result: RunResult) -> list[tuple[str, str]]:
    """Report header entries: the run identity and the policies in effect."""
    gt = result.ground_truth
    align = result.alignment
    conf = result.confidence_policy
    entries: list[tuple[str, str]] = [
        ("method", result.method or "(unnamed)"),
        (
            "dataset",
            f"{result.dataset.dataset} / {result.dataset.variant} "
            f"(split: {result.split or 'n/a'})",
        ),
        ("protocol", f"{result.protocol} v{result.protocol_version}"),
        ("protocol hash", result.protocol_hash),
        ("fidelity", result.fidelity),
        (
            "ground truth",
            f"modality={gt.modality}, provenance={gt.provenance}, "
            f"independence={gt.independence}, density={gt.density}",
        ),
        ("local evaluation", result.local_evaluation.status),
        (
            "alignment",
            f"mode={align.mode}, estimate_on={align.estimate_on}, "
            f"solver={align.solver}, granularity={align.granularity}",
        ),
        ("confidence policy", conf.policy),
        (
            "sampling",
            f"pred: {_sampling_summary(result.sampling.pred)}; "
            f"gt: {_sampling_summary(result.sampling.gt)}",
        ),
        ("failure policy", result.failure_policy.policy),
        ("backends", _backend_summary(result)),
        ("eval3r version", result.eval3r_version),
        ("timestamp", result.timestamp),
    ]
    if result.command:
        entries.append(("command", result.command))
    return entries


def _aggregate_rows(result: RunResult) -> tuple[list[str], list[list[str]]]:
    columns = ["metric", "value", "unit", "statistic", "threshold"]
    specs = {spec.name: spec for spec in result.metric_definitions}
    units: dict[str, str] = {}
    for mr in result.per_scene_metrics:
        if mr.unit is not None and mr.name not in units:
            units[mr.name] = mr.unit
    rows: list[list[str]] = []
    for name, value in result.metrics.items():
        spec = specs.get(name)
        threshold = spec.threshold if spec is not None else None
        rows.append(
            [
                name,
                format_value(value),
                units.get(name, ""),
                (spec.statistic or "") if spec is not None else "",
                format_value(threshold) if threshold is not None else "",
            ]
        )
    return columns, rows


def _row_key(mr: MetricResult) -> tuple[str, str]:
    return (mr.scene_id or "<unknown>", mr.frame_id or "")


def _per_scene_rows(result: RunResult) -> tuple[list[str], list[list[str]]]:
    metric_names = [spec.name for spec in result.metric_definitions]
    for mr in result.per_scene_metrics:  # metrics outside the spec list, kept visible
        if mr.name not in metric_names:
            metric_names.append(mr.name)
    columns = ["scene", "status", *metric_names, "failure reason"]

    rows: dict[tuple[str, str], dict[str, str]] = {}
    for mr in result.per_scene_metrics:
        row = rows.setdefault(_row_key(mr), {"status": "ok"})
        row[mr.name] = format_value(mr.value)
    for failure in result.failed_scenes:
        row = rows.setdefault((failure.scene_id, ""), {})
        row["status"] = f"failed ({failure.stage})"
        row["failure reason"] = failure.reason

    out: list[list[str]] = []
    for (scene_id, frame_id), row in sorted(rows.items()):
        label = f"{scene_id}/{frame_id}" if frame_id else scene_id
        out.append(
            [
                label,
                row.get("status", ""),
                *[row.get(name, "") for name in metric_names],
                row.get("failure reason", ""),
            ]
        )
    return columns, out


def build_report_data(result: RunResult) -> ReportData:
    """Reduce a ``RunResult`` to renderer-ready tables (no metric recomputation)."""
    coverage = CoverageInfo(
        expected=result.n_scenes_expected,
        evaluated=result.n_scenes_evaluated,
        failed=len(result.failed_scenes),
        failure_policy=result.failure_policy.policy,
    )
    aggregate_columns, aggregate_rows = _aggregate_rows(result)
    per_scene_columns, per_scene_rows = _per_scene_rows(result)
    method = result.method or "(unnamed method)"
    title = f"eval3r report: {method} on {result.dataset.dataset} ({result.protocol})"
    return ReportData(
        title=title,
        header=build_header(result),
        coverage=coverage,
        banner=coverage_banner(coverage),
        aggregate_columns=aggregate_columns,
        aggregate_rows=aggregate_rows,
        per_scene_columns=per_scene_columns,
        per_scene_rows=per_scene_rows,
        failures=[(f.scene_id, f.stage, f.reason) for f in result.failed_scenes],
    )
