# Task 029: Multithreaded Benchmark Execution

## Goal

Add opt-in scene-level multithreading to dataset benchmark runs so independent scenes can
be evaluated concurrently without changing scores, protocol identity, coverage accounting,
failure visibility, or deterministic result serialization.

## Scope

- Add a positive `workers` run option to the benchmark Python API, with a default of `1`.
- Add `--workers` to `e3r benchmark run`, also defaulting to `1`.
- Evaluate independent scenes with a thread pool when `workers > 1`.
- Support every existing geometry benchmark path:
  - eval3r-native geometry evaluation;
  - ScanNet-style visibility rendering and TSDF culling;
  - DTU point-array official-like evaluation;
  - Tanks and Temples official-toolbox subprocess evaluation;
  - ETH3D official-tool subprocess evaluation.
- Preserve deterministic split-order result assembly and aggregation regardless of task
  completion order.
- Preserve rich per-scene progress, full failure reasons, and all protocol failure policies.
- Record requested and effective worker counts and executor type in run configuration and
  result metadata.
- Serialize pyrender/EGL visibility-culling calls with a module-level lock while allowing the
  other stages of different culled scenes to overlap.
- Document concurrency behavior, backend safety expectations, expected GIL-dependent speedup,
  and per-worker memory costs.

## Out of Scope

- Frame-level parallelism within a scene or depth sequence.
- Multiprocessing, distributed execution, cluster schedulers, or GPU assignment.
- Making concurrent execution the default.
- Changing metric formulas, aggregation rules, sampling seeds, protocol fields, protocol
  hashes, or official-toolbox scoring behavior.
- Parallelizing `e3r benchmark validate` unless it becomes a necessary shared helper during
  implementation; if changed, its behavior and tests must be documented explicitly.

## Relevant Files

- `.agent/plan.md`
- `.agent/schema.md`
- `.agent/protocols.md`
- `.agent/backends.md`
- `.agent/reproducibility.md`
- `eval3r/api.py`
- `eval3r/pipeline/benchmark.py`
- `eval3r/cli/benchmark.py`
- `eval3r/backends/visibility_render.py`
- `eval3r/backends/tnt_official.py`
- `eval3r/backends/eth3d_official.py`
- `tests/unit/test_benchmark_runner.py`
- `tests/integration/test_benchmark_fixture.py`
- Existing integration tests for ScanNet, DTU, Tanks and Temples, and ETH3D benchmarks

## Plan

1. Audit the current per-scene execution paths and shared objects before editing.
   - Identify mutable adapter state established by `bind_protocol()` and confirm that all
     subsequent scene reads are safe concurrently or protect the mutable access explicitly.
   - Inspect backend instances for mutable state, renderer/context ownership, temporary-file
     naming, subprocess output paths, and thread-safety guarantees.
   - Ensure every official-toolbox invocation retains a unique per-scene output directory.
   - Inspect the real Tanks and Temples `run.py` before permitting concurrent invocations:
     although eval3r launches it with `cwd` set to the toolbox directory, verify that the
     script writes all result-affecting and temporary outputs beneath its supplied `--out-dir`
     and does not collide through cwd-relative files.
   - Account for memory multiplication explicitly: N concurrent scenes can retain N loaded
     meshes/point clouds, N official-toolbox subprocess working sets (including ICP), and N
     debug distance captures when reporting requests them.

2. Define and validate the public run configuration.
   - Add `workers: int = 1` to `eval3r.api.run_benchmark` and
     `run_benchmark_geometry`, threading it through without altering the protocol object.
   - Add `--workers` to `e3r benchmark run` with clear help text describing scene-level
     threads, opt-in behavior, and the serial default. The help text must also set expectations
     that threads work best for official subprocess, I/O, and native KD-tree/Open3D workloads
     and are not a linear CPU multiplier for GIL-bound stages.
   - Reject booleans, zero, and negative values with an error that names the invalid value and
     states that `workers` must be a positive integer.
   - Define the effective worker count as `min(workers, number_of_scenes)`, while retaining
     `1` for an empty split if empty splits are already accepted. Do not silently derive a CPU
     count.

3. Extract one side-effect-contained scene evaluation callable.
   - Route native, visibility-culling, DTU, Tanks and Temples, and ETH3D modes through the
     same dispatcher that returns a complete `SceneOutcome`.
   - Give each task all immutable inputs it needs; do not mutate shared result lists from
     worker threads.
   - Add a module-level lock for pyrender/EGL visibility culling and hold it only around
     `vis_backend.cull()`. `RenderTsdfVisibilityCull.cull()` creates a
     `pyrender.OffscreenRenderer` per call, and concurrent EGL context creation is
     driver-dependent and may segfault the process, bypassing every eval3r failure policy.
     Loading, normalization, sampling, nearest-neighbor queries, and metric computation for
     other culled scenes may continue concurrently outside this lock.
   - Isolate or lock any other backend operation found not to be thread-safe. Prefer per-task
     or thread-local state when construction is safe; use a narrow lock when a backend
     resource inherently cannot run concurrently, and record that limitation in Findings.
   - Do not clone or reconstruct a caller-supplied adapter/registry unless their public
     interfaces explicitly support it.

4. Implement deterministic orchestration.
   - Keep the existing direct loop for `workers == 1` so serial behavior and tracebacks remain
     compatible.
   - For `workers > 1`, use `concurrent.futures.ThreadPoolExecutor` for scene tasks.
   - Drain completed futures from the orchestrating caller thread with `as_completed` and
     invoke every progress callback from that same thread. Worker threads must never call
     `progress`; Rich CLI output and user-supplied callbacks therefore do not need to be
     thread-safe. Report progress in completion order, with every callback continuing to
     receive the matching scene and full outcome.
   - Store outcomes by original split index, then apply failure policy and assemble metrics,
     failures, alignment transforms, debug captures, and visualization captures strictly in
     split order.
   - Keep sampling deterministic: scene-derived seeds must depend only on the recorded base
     seed and scene identifier, never scheduling or completion order.

5. Preserve explicit failure-policy semantics.
   - `skip_and_flag`: finish all submitted scenes, retain each failure reason, and aggregate
     successful scenes in split order.
   - `score_worst`: finish all submitted scenes and inject worst values in split order exactly
     as the serial runner does.
   - `abort`: on the first failed completed future, stop submitting new work if submission is
     bounded, cancel futures that have not started, safely join already-running tasks, and
     raise `SceneEvaluationError` with the failed scene, stage, and complete reason. Work that
     was already running may finish, but its result must not be included or written. With
     `workers > 1`, the raised failure is intentionally the first failure observed in
     completion order and is therefore scheduling-dependent when multiple scenes fail; it is
     not necessarily the earliest failing scene in split order. `workers == 1` retains the
     current exact split-order abort semantics.
   - Avoid an unbounded submit-all design when practical so `abort` can prevent unnecessary
     scene/toolbox work; keep no more than the effective worker count in flight.

6. Record execution configuration without changing protocol identity.
   - Add `workers_requested`, `workers_effective`, and `executor: thread` (or `serial` when
     effective workers are one) to `config.yaml` and `RunResult.metadata`.
   - Echo requested/effective workers in the resolved CLI configuration before writing the
     run directory.
   - Do not add worker settings to `EvalProtocol`, protocol YAML, or canonical protocol
     hashing because scheduling does not define scoring semantics.

7. Update documentation in the same change.
   - Document the CLI and Python API options, serial default, scene-level granularity,
     deterministic output ordering, caller-thread progress callbacks, completion-order
     progress, and the concurrent abort nondeterminism described above.
   - Explain that threads provide the largest wall-clock gains for independent official
     subprocesses (Tanks and Temples and ETH3D), I/O-heavy work, and native operations that
     release the GIL such as scipy cKDTree and Open3D. DTU's Python port and pure-NumPy/Python
     stages can remain GIL-bound, so `--workers` is not a linear CPU multiplier.
   - Document that pyrender/EGL culling is deliberately serialized for process safety, so
     ScanNet visibility-culled runs receive limited speedup even though their non-rendering
     stages may overlap.
   - Add a memory caution: concurrent scenes multiply loaded geometry, subprocess/ICP working
     sets, and optional per-scene debug distance captures; users should choose workers to fit
     available RAM.
   - Update `.agent/reproducibility.md` to identify concurrency settings as recorded run
     configuration rather than protocol state.
   - Update `.agent/backends.md` with any backend-specific serialization, resource isolation,
     or real-tool limitations found during the audit.

8. Add behavior-focused tests.
   - Validate accepted and rejected worker counts at API and CLI boundaries.
   - Prove `workers=1` retains current results and ordering.
   - Use a controlled native scene evaluator with synchronization events/barriers to prove
     that at least two scene tasks overlap when `workers > 1`; do not use timing-only
     assertions.
   - Force out-of-order completion and assert deterministic split-order metrics, failures,
     debug/alignment records, and serialized results.
   - Assert progress is delivered in completion order and always carries the matching scene
     outcome, and assert callbacks execute on the caller thread rather than a worker thread.
   - Cover `abort`, `skip_and_flag`, and `score_worst`, including cancellation/bounded
     scheduling behavior for abort.
   - Compare serial and threaded native integration runs for identical metrics, coverage,
     protocol hash, and run-directory result content apart from concurrency metadata and
     timestamps.
   - Exercise the real visibility backend with tiny public fixtures where supported.
     Do not create a test that attempts unlocked concurrent EGL context creation; test the
     serialization boundary without bypassing the module-level lock.
   - Official-wrapper concurrency tests must invoke the real configured DTU/Tanks and
     Temples/ETH3D evaluator paths. When a required external toolbox is absent, skip cleanly
     with its exact missing-tool reason; never substitute a fake official evaluator.

9. Verify the completed slice and record exact outcomes below.
   - Run focused benchmark runner and CLI tests first.
   - Run the existing dataset benchmark integration tests.
   - Run `ruff check .`, `pytest`, the configured type checker (`mypy eval3r` or the actual
     project equivalent), and `mkdocs build`.
   - Run real official-toolbox benchmark checks when configured, recording tool sources,
     versions/commits, commands, skips, and any mechanical compatibility patch.

## Findings

- The current benchmark runner evaluates scenes serially in split order inside
  `run_benchmark_geometry`.
- Native, visibility-culling, and three official evaluator input modes share that loop and
  therefore need one common concurrency contract.
- `RenderTsdfVisibilityCull.cull()` constructs a pyrender offscreen EGL context per call;
  concurrent context creation can segfault at the driver level, so culling itself must be
  serialized even when surrounding scene stages run concurrently.
- Expected speedups are workload-dependent because Python threads rely on subprocess work,
  I/O, or native operations releasing the GIL. Pure-Python/NumPy-heavy stages may see little
  benefit.
- Concurrent workers multiply resident geometry, debug-distance, and official-tool subprocess
  memory, so worker selection must account for available RAM.
- Task 028 remains the active known-good validation slice; this task is queued after it.
- Further backend thread-safety findings must be recorded here during implementation rather
  than assumed.

## Decisions

- Multithreading is opt-in and defaults to one worker.
- Parallelism is across scenes, not within scene evaluation stages.
- All currently supported benchmark evaluator paths are in scope.
- Result assembly and aggregation remain deterministic in dataset split order; only progress
  output follows completion order.
- Progress callbacks always run on the orchestrating caller thread, never worker threads.
- Pyrender/EGL visibility-culling calls are protected by a module-level lock; only the cull
  call is serialized, leaving other per-scene stages eligible to overlap.
- Concurrent `abort` raises the first failure observed in completion order and can therefore
  vary when several scenes fail; serial `workers == 1` preserves split-order failure choice.
- Worker settings are reproducibility metadata and run configuration, not protocol state, so
  they do not affect the protocol hash.
- Real official evaluators remain mandatory for official-path verification; fake or lookalike
  evaluators are forbidden.

## Verification

Not run; task is queued. Record exact commands, pass/fail results, real official-toolbox runs
or explicit skips, and any result-neutral compatibility patches here during implementation.

## Status

todo
