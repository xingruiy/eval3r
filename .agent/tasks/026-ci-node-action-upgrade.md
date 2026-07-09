# Goal

Remove the GitHub Actions Node.js 20 deprecation warning from CI.

# Scope

- Update first-party GitHub Actions workflow pins that currently use Node 20 actions.
- Keep the existing CI job behavior unchanged.

# Out of Scope

- Changing the Python version matrix.
- Changing lint, type-check, test, or docs commands.

# Relevant Files

- `.github/workflows/ci.yml`

# Plan

1. Identify actions emitting the warning.
2. Verify current replacement action versions run on Node 24.
3. Update workflow pins.
4. Validate workflow YAML shape locally by inspection.

# Findings

- CI uses `actions/checkout@v4` and `actions/setup-python@v5`.
- Current official action metadata reports `actions/checkout@v7` and
  `actions/setup-python@v6` with `runs.using: node24`.

# Decisions

- Use the current major tags for the first-party actions: `checkout@v7` and
  `setup-python@v6`.

# Verification

```bash
python - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/ci.yml')
data = yaml.safe_load(path.read_text(encoding='utf-8'))
steps = data['jobs']['check']['steps']
uses = [step.get('uses') for step in steps if 'uses' in step]
print('\n'.join(uses))
PY
# actions/checkout@v7
# actions/setup-python@v6
```

The deprecation warning itself is emitted by GitHub-hosted runners, so final
confirmation happens on the next CI run.

# Status

done
