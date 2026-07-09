"""Task 006 tests: run-directory completeness, required fields, stable CSV columns."""

from __future__ import annotations

import csv
import json

from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.core.schema import (
    AggregationSpec,
    AlignmentSpec,
    ConfidenceSpec,
    DatasetVariant,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    MaskingSpec,
    MetricSpec,
    SamplingSideSpec,
    SamplingSpec,
)
from eval3r.reports import PER_SCENE_COLUMNS, read_run_result_json, write_run_directory

REQUIRED_RESULTS_JSON_FIELDS = [
    "schema_version",
    "protocol",
    "protocol_hash",
    "dataset",
    "split",
    "ground_truth",
    "local_evaluation",
    "n_scenes_expected",
    "n_scenes_evaluated",
    "failure_policy",
    "metrics",
    "metric_definitions",
    "alignment",
    "masking",
    "sampling",
    "aggregation",
    "backend_versions",
    "command",
    "environment",
]


def _run_result() -> RunResult:
    gt = GroundTruthSpec(
        modality="pointcloud",
        provenance="laser_scan",
        independence="independent",
        density="dense_surface",
    )
    return RunResult(
        schema_version=1,
        eval3r_version="0.3.0",
        method="example_method",
        method_version="0.1.0",
        dataset=DatasetVariant(dataset="dtu", variant="mvs_2014", split="test"),
        split="test",
        protocol="dtu_native_pointcloud",
        protocol_version="0.1.0",
        protocol_hash="sha256:abc",
        fidelity="native",
        ground_truth=gt,
        local_evaluation=LocalEvaluationSpec(status="supported"),
        n_scenes_expected=2,
        n_scenes_evaluated=1,
        failed_scenes=[
            SceneFailure(scene_id="scan002", stage="load", reason="prediction file missing")
        ],
        failure_policy=FailurePolicySpec(policy="skip_and_flag"),
        metrics={"accuracy": 0.03, "fscore": 0.61},
        metric_definitions=[MetricSpec(name="fscore", threshold=0.05)],
        per_scene_metrics=[
            MetricResult(
                name="accuracy",
                value=0.03,
                unit="m",
                statistic="mean",
                scene_id="scan001",
                protocol="dtu_native_pointcloud",
                protocol_hash="sha256:abc",
                backend="scipy",
                n_points_pred=200000,
                n_points_gt=180000,
                valid_fraction=0.99,
            ),
            MetricResult(
                name="fscore",
                value=0.61,
                scene_id="scan001",
                protocol="dtu_native_pointcloud",
                protocol_hash="sha256:abc",
            ),
        ],
        confidence_policy=ConfidenceSpec(),
        alignment=AlignmentSpec(mode="none"),
        masking=MaskingSpec(),
        sampling=SamplingSpec(pred=SamplingSideSpec(), gt=SamplingSideSpec()),
        aggregation=AggregationSpec(),
        backend_versions={
            "nearest_neighbor": {"name": "scipy", "library": "scipy", "version": "1.15.3"},
            "pointcloud": {"name": "plyfile", "library": "plyfile", "version": "1.0"},
        },
        environment={"python_version": "3.10.12", "os": "Linux"},
        command="e3r benchmark run preds/ --dataset dtu",
        timestamp="2026-07-04T00:00:00Z",
    )


def test_run_directory_has_core_files(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    for name in (
        "results.json",
        "results.csv",
        "per_scene.csv",
        "failures.json",
        "environment.json",
        "backend_versions.json",
        "logs.txt",
    ):
        assert (out / name).is_file(), f"missing {name}"


def test_results_json_round_trips_and_has_required_fields(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    data = json.loads((out / "results.json").read_text())
    for field in REQUIRED_RESULTS_JSON_FIELDS:
        assert field in data, f"results.json missing {field}"
    # round-trips back into a validated model
    assert read_run_result_json(out / "results.json").method == "example_method"


def test_failures_json_records_partial_coverage(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    data = json.loads((out / "failures.json").read_text())
    assert data["failure_policy"] == "skip_and_flag"
    assert data["n_scenes_failed"] == 1
    assert data["failed_scenes"][0]["scene_id"] == "scan002"
    assert data["failed_scenes"][0]["policy_action"] == "skip_and_flag"


def test_per_scene_csv_has_stable_columns_and_both_scenes(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    with (out / "per_scene.csv").open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == PER_SCENE_COLUMNS
        rows = {r["scene_id"]: r for r in reader}
    assert set(rows) == {"scan001", "scan002"}
    assert rows["scan001"]["status"] == "ok"
    assert rows["scan001"]["accuracy"] == "0.03"
    assert rows["scan001"]["backend_nn"] == "scipy"
    assert rows["scan002"]["status"] == "failed"
    assert "missing" in rows["scan002"]["failure_reason"]


def test_results_csv_lists_aggregate_metrics(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    with (out / "results.csv").open() as fh:
        rows = {r["metric"]: r["value"] for r in csv.DictReader(fh)}
    assert rows["accuracy"] == "0.03"
    assert rows["fscore"] == "0.61"


def test_conditional_files_written_when_provided(tmp_path) -> None:
    out = write_run_directory(
        _run_result(),
        tmp_path / "run",
        manifest={"method": "example_method"},
        config={"overrides": {}},
        alignment_transforms=[{"scene_id": "scan001", "mode": "none"}],
        logs="ran ok\n",
    )
    assert (out / "manifest.yaml").is_file()
    assert (out / "config.yaml").is_file()
    assert (out / "alignment_transforms.json").is_file()
    assert (out / "logs.txt").read_text() == "ran ok\n"


def test_conditional_files_absent_when_not_provided(tmp_path) -> None:
    out = write_run_directory(_run_result(), tmp_path / "run")
    assert not (out / "manifest.yaml").exists()
    assert not (out / "config.yaml").exists()
    assert not (out / "alignment_transforms.json").exists()


def test_protocol_copy_from_builtin(tmp_path) -> None:
    from eval3r.protocols import load_protocol

    proto = load_protocol("single_geometry")
    out = write_run_directory(_run_result(), tmp_path / "run", protocol=proto)
    assert (out / "protocol.yaml").is_file()
    import yaml

    reloaded = yaml.safe_load((out / "protocol.yaml").read_text())
    assert reloaded["name"] == "single_geometry"
