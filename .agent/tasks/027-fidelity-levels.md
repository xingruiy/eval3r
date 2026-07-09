Goal
----
Collapse top-level fidelity levels to exactly `official`, `native`, and `server`,
and rename built-in protocol IDs that carried the old fidelity names.

Scope
-----
- Update schema/type docs and Pydantic literal types.
- Rename affected built-in protocol files and their `name` fields.
- Update built-in protocol fidelity values, fixtures, tests, and expected hashes.
- Update user-facing docs and durable task index references.

Out of Scope
------------
- Renaming non-fidelity `server_only` enum values for local-evaluation or
  ground-truth status.
- Adding compatibility aliases for old protocol IDs.
- Changing evaluator behavior or metric semantics.

Relevant Files
--------------
- `.agent/schema.md`
- `.agent/protocols.md`
- `.agent/reproducibility.md`
- `eval3r/core/types.py`
- `eval3r/protocols/builtin/*.yaml`
- `tests/`

Plan
----
- Replace `Fidelity` literals with `official`, `native`, and `server`.
- Rename the DTU official-like protocol ID to `dtu_native_pointcloud`.
- Rename the Tanks and Temples server-only protocol ID to
  `tanks_temples_intermediate_server`.
- Replace protocol fidelity values and references in docs, fixtures, and tests.
- Recompute expected protocol hashes.
- Run focused tests and record outcomes.

Findings
--------
- `Fidelity` is now exactly `official`, `native`, and `server`.
- Built-in protocol names now use `dtu_native_pointcloud` and
  `tanks_temples_intermediate_server`.
- Existing native protocols serialize `fidelity: native`; official toolbox wrappers
  remain `fidelity: official`; withheld-server protocol stubs serialize
  `fidelity: server`.
- DTU still records `official_eval: dtu` and evaluator method
  `validated_official_port`; only the top-level fidelity and public protocol ID changed.

Decisions
---------
- DTU's validated official-like evaluator remains operationally unchanged, but
  the protocol's top-level fidelity is `native`.
- The old protocol IDs are not kept as aliases.
- Non-fidelity `server_only` values remain unchanged.

Verification
------------
- `PYTHONPATH=. pytest tests/unit/test_schema.py tests/unit/test_protocols.py tests/unit/test_hashing.py`
  passed: 76 passed.
- `PYTHONPATH=. /home/xingrui/miniconda3/envs/dl/bin/python -m pytest tests/unit/test_result_writer.py tests/unit/test_reports.py tests/unit/test_diff.py`
  passed: 46 passed.
- `PYTHONPATH=. /home/xingrui/miniconda3/envs/dl/bin/python -m pytest tests/integration/test_dtu_native_benchmark.py tests/integration/test_dtu_adapter_benchmark.py tests/integration/test_scannet_benchmark.py tests/integration/test_neural_rgbd_benchmark.py tests/integration/test_tnt_benchmark.py tests/integration/test_eth3d_benchmark.py`
  passed: 15 passed, 2 skipped because real official external tools were not configured.
- `PYTHONPATH=. /home/xingrui/miniconda3/envs/dl/bin/python -m pytest`
  passed: 541 passed, 6 skipped because real official external tools were not configured.
- The default `/usr/bin/python` environment could not collect report tests because
  Matplotlib requires NumPy >= 1.23 while that interpreter has NumPy 1.21.5; the
  `dl` conda env was used for full verification.

Status
------
done
