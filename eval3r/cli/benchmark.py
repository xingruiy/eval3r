"""`e3r benchmark` command group.

`run` executes the generic benchmark loop (task 008) across a dataset split;
`validate` runs the same preflight and prediction resolution *without* computing
metrics. Both are verbose by design (``rich``): they echo the resolved protocol
(name + hash), dataset/split, the alignment / masking / sampling / confidence /
failure policies in effect, the resolved paths, per-scene outcomes, and — never
hidden behind an average — partial coverage (CLAUDE.md CLI verbosity rules).
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.core.environment import capture_environment
from eval3r.core.errors import Eval3rError
from eval3r.core.registry import default_registry as default_backend_registry
from eval3r.datasets import default_registry as default_dataset_registry
from eval3r.pipeline.benchmark import (
    BenchmarkRunOutput,
    load_or_infer_manifest,
    preflight,
    run_benchmark_geometry,
)
from eval3r.pipeline.runner import SceneOutcome
from eval3r.pipeline.stages.sample import DEFAULT_BASE_SEED
from eval3r.protocols import load_protocol
from eval3r.reports.alignment_vis import (
    write_alignment_vis_outputs,
    write_trajectory_alignment_vis_outputs,
)
from eval3r.reports.plots import write_geometry_debug_outputs
from eval3r.reports.run_directory import default_run_dir_name, write_run_directory

benchmark_app = typer.Typer(
    help="Run or validate a dataset benchmark under a named protocol.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _echo_config(run: BenchmarkRunOutput, pred_root: Path, out_dir: Path) -> None:
    proto = run.protocol
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("protocol", f"{proto.name}  ({run.protocol_hash})")
    table.add_row("dataset / split", f"{run.config['dataset']} / {run.config['split']}")
    table.add_row("prediction root", str(pred_root))
    table.add_row("output dir", str(out_dir))
    table.add_row("scenes", str(run.config["n_scenes"]))
    table.add_row("manifest", "inferred" if run.manifest_inferred else "declared")
    table.add_row("alignment", f"mode={proto.alignment.mode} solver={proto.alignment.solver}")
    if run.result.adaptation is not None:
        adaptation = run.result.adaptation
        table.add_row(
            "adaptation",
            f"source={adaptation.source} reason={adaptation.reason} "
            f"alignment={adaptation.alignment} "
            f"[{'transformed' if adaptation.transformed else 'passthrough'}]",
        )
    table.add_row(
        "masking",
        f"pred={proto.masking.pred_culling.method} gt={proto.masking.gt_culling.method}",
    )
    table.add_row(
        "sampling",
        f"pred={proto.sampling.pred.method}/{proto.sampling.pred.n_points} "
        f"gt={proto.sampling.gt.method}/{proto.sampling.gt.n_points}",
    )
    table.add_row("failure policy", proto.failure_policy.policy)
    console.print(Panel(table, title="resolved benchmark configuration", expand=False))


def _echo_summary(run: BenchmarkRunOutput, out_dir: Path) -> None:
    result = run.result
    if result.failed_scenes:
        console.print(
            f"[yellow]partial coverage:[/] {result.n_scenes_evaluated}/"
            f"{result.n_scenes_expected} scenes evaluated, {len(result.failed_scenes)} failed "
            f"(policy: {result.failure_policy.policy})"
        )
    else:
        console.print(
            f"[green]all scenes evaluated:[/] {result.n_scenes_evaluated}/"
            f"{result.n_scenes_expected}"
        )
    metrics = Table(title="aggregate metrics (per_scene_then_mean)", header_style="bold")
    metrics.add_column("metric")
    metrics.add_column("value", justify="right")
    for name, value in result.metrics.items():
        metrics.add_row(name, "-" if value is None else f"{value:.6g}")
    console.print(metrics)
    emitted = []
    if run.debug_scenes:
        emitted.append("geometry errors/histograms")
    if run.alignment_vis:
        emitted.append("geometry alignment")
    if run.trajectory_alignment_vis:
        emitted.append("trajectory alignment")
    if emitted:
        debug = Table(show_header=False, box=None, pad_edge=False)
        debug.add_column(style="bold cyan")
        debug.add_column()
        debug.add_row("scene debug root", str(out_dir / "debug" / "scenes"))
        debug.add_row("emitted", ", ".join(emitted))
        debug.add_row("indexes", "debug/debug_index.json")
        console.print(Panel(debug, title="debug artifacts", expand=False))
    console.print(f"[green]run directory written:[/] {out_dir}")


def _print_scene(scene_id: str, outcome: SceneOutcome) -> None:
    if outcome.failure is None:
        console.print(f"  [green]ok[/]   {scene_id}")
    else:
        err_console.print(
            f"  [red]fail[/] {scene_id} (stage {outcome.failure.stage}): {outcome.failure.reason}"
        )


@benchmark_app.command("run")
def benchmark_run(
    pred_root: Path = typer.Argument(..., help="Directory of predictions (per-scene files)."),
    dataset: str = typer.Option(..., "--dataset", help="Registered dataset adapter name."),
    split: str = typer.Option(..., "--split", help="Dataset split to evaluate."),
    protocol: str = typer.Option(..., "--protocol", help="Built-in protocol name or YAML path."),
    root: Path | None = typer.Option(
        None, "--root", help="Dataset root the adapter needs (e.g. ground-truth files)."
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Prediction manifest YAML (else pred_root/manifest.yaml/inferred)."
    ),
    method: str | None = typer.Option(None, "--method", help="Method name recorded in the result."),
    adapt: str | None = typer.Option(
        None, "--as", "--adapt", help="Prediction adaptation override, e.g. opengl@sim3."
    ),
    out: Path | None = typer.Option(None, "--out", help="Run directory to write."),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help=(
            "Run-config base seed for protocols with 'derive' sampling seeds; repeat a "
            "run with different values to measure sampling sensitivity. Recorded in the "
            "run config and result metadata. Default keeps the documented base seed."
        ),
    ),
) -> None:
    """Run a dataset benchmark under a named protocol."""
    command = "e3r " + shlex.join(sys.argv[1:]) if len(sys.argv) > 1 else "e3r benchmark run"

    try:
        proto = load_protocol(protocol)
        adapter = default_dataset_registry().create(dataset, root)
        run = run_benchmark_geometry(
            pred_root, adapter, proto, split,
            manifest_path=manifest, command=command,
            environment=capture_environment(command=command), method=method,
            adapt=adapt,
            progress=_print_scene,
            base_seed=DEFAULT_BASE_SEED if seed is None else seed,
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="benchmark failed", style="red", expand=False))
        err_console.print_exception()
        raise typer.Exit(code=1) from exc

    out_dir = (
        Path(out)
        if out is not None
        else Path("runs") / default_run_dir_name(run.result, now=datetime.now(timezone.utc))
    )
    _echo_config(run, Path(pred_root), out_dir)
    write_run_directory(
        run.result, out_dir,
        protocol=run.protocol, manifest=run.manifest, config=run.config,
        environment=run.result.environment, backend_versions=run.result.backend_versions,
        alignment_transforms=run.alignment_transforms,
    )
    pointcloud_backend = default_backend_registry().require(
        "pointcloud", run.protocol.backend_preferences.get("pointcloud", "plyfile")
    )
    write_geometry_debug_outputs(
        run.debug_scenes, out_dir,
        reporting=run.protocol.reporting,
        pointcloud_backend=pointcloud_backend,
    )
    write_alignment_vis_outputs(
        run.alignment_vis, out_dir,
        pointcloud_backend=pointcloud_backend,
    )
    write_trajectory_alignment_vis_outputs(
        run.trajectory_alignment_vis, out_dir,
        pointcloud_backend=pointcloud_backend,
    )
    _echo_summary(run, out_dir)


@benchmark_app.command("validate")
def benchmark_validate(
    pred_root: Path = typer.Argument(..., help="Directory of predictions (per-scene files)."),
    dataset: str = typer.Option(..., "--dataset", help="Registered dataset adapter name."),
    split: str = typer.Option(..., "--split", help="Dataset split to evaluate."),
    protocol: str = typer.Option(..., "--protocol", help="Built-in protocol name or YAML path."),
    root: Path | None = typer.Option(None, "--root", help="Dataset root the adapter needs."),
    manifest: Path | None = typer.Option(None, "--manifest", help="Prediction manifest YAML."),
) -> None:
    """Validate preflight + prediction resolution for every scene, without evaluating."""
    try:
        proto = load_protocol(protocol)
        adapter = default_dataset_registry().create(dataset, root)
        preflight(adapter, split, proto)
        scenes = list(adapter.iter_scenes(split))
        pred_manifest, _, inferred = load_or_infer_manifest(
            Path(pred_root), scenes, proto, manifest_path=manifest
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="validation failed", style="red", expand=False))
        raise typer.Exit(code=1) from exc

    ok = 0
    missing: list[tuple[str, str]] = []
    for scene_id in scenes:
        try:
            adapter.resolve_prediction(Path(pred_root), scene_id, pred_manifest)
            adapter.load_scene(scene_id)
            ok += 1
        except Eval3rError as exc:
            missing.append((scene_id, str(exc)))

    console.print(
        f"protocol [bold]{proto.name}[/] ({'inferred' if inferred else 'declared'} manifest): "
        f"{ok}/{len(scenes)} scenes resolve"
    )
    for scene_id, reason in missing:
        err_console.print(f"  [red]missing[/] {scene_id}: {reason}")
    if missing:
        raise typer.Exit(code=1)
