# Goal

Fix the CI failure in `tests/integration/test_prediction_cli.py::test_cli_validate_fails_without_manifest`.

# Scope

- Preserve verbose prediction validation errors.
- Make the missing-manifest reason stable under Rich wrapping with long temporary paths.
- Verify the targeted integration test.

# Out of Scope

- Prediction schema changes.
- Prediction manifest layout changes.
- Dataset, metric, or protocol behavior changes.

# Relevant Files

- `eval3r/predictions/reader.py`
- `tests/integration/test_prediction_cli.py`

# Plan

1. Reproduce the missing substring with a long temp path and narrow CLI width.
2. Adjust the missing-manifest message so the actionable phrase is not split by the path.
3. Update the integration test to cover the narrow-width rendering.
4. Run the targeted prediction CLI tests.

# Findings

- `load_prediction_manifest` raised a message containing `has no manifest.yaml`.
- Rich wraps the long temp path between `no` and `manifest.yaml`, so the literal substring
  `no manifest.yaml` can disappear from `CliRunner` output even though the message is present.

# Decisions

- Start the missing-manifest error with `no manifest.yaml for prediction root ...`.
  This keeps the user-facing reason visible and grep-able before the long path.

# Verification

```bash
PYTHONPATH=/home/xingrui/xingrui_ws/codes/eval3r pytest tests/integration/test_prediction_cli.py -q
# 5 passed, 1 warning

PYTHONPATH=/home/xingrui/xingrui_ws/codes/eval3r pytest tests/unit/test_prediction_reader.py tests/integration/test_prediction_cli.py -q
# 11 passed, 1 warning
```

Initial local `pytest tests/integration/test_prediction_cli.py -q` failed before collection
because the user-site/ROS pytest plugin stack did not import the in-tree `eval3r` package.
The reruns above set `PYTHONPATH` to the repository root.

# Status

done
