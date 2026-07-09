# 024 — Fusion policy documentation

## Goal

Remove blanket prohibitions on TSDF fusion, RGB-D fusion, volumetric fusion,
online mapping, SLAM-style wording, and similar reconstruction-pipeline language
from the design files and repeated documentation.

## Scope

- Update source-of-truth design docs so fusion/integration support is no longer
  categorically forbidden.
- State that any required TSDF/RGB-D/depth integration path should delegate the
  commodity operation to established backends such as Open3D.
- Keep explicit protocol control, metadata recording, and non-silent behavior.
- Update public docs and Python docstrings/comments that repeat the old
  prohibition.

## Out of Scope

- No implementation of a new fusion pipeline.
- No schema, protocol hash, metric formula, or result-field changes.
- No change to official-toolbox behavior.
- No relaxation of metadata or explicit-protocol requirements.

## Relevant Files

- `CLAUDE.md`
- `.agent/plan.md`
- `.agent/backends.md`
- `.agent/metrics.md`
- `.agent/tasks/011-scannet-adapter.md`
- `.agent/tasks/014-depth-metrics.md`
- `.agent/tasks/README.md`
- `docs/index.md`
- `docs/backends.md`
- `docs/metrics.md`
- `eval3r/metrics/depth.py`
- `eval3r/pipeline/depth_runner.py`

## Plan

1. Replace hard-boundary/non-goal wording with explicit delegated-support
   wording.
2. Update public docs and Python docstrings to remove repeated blanket bans.
3. Search for remaining old prohibition wording.
4. Run `mkdocs build`.

## Findings

- Blanket prohibition wording existed in `CLAUDE.md`, `.agent/plan.md`,
  `.agent/backends.md`, `.agent/metrics.md`, public docs, two Python docstrings,
  and stale completed task notes.
- No schema fields, protocol hashes, metric formulas, or result writers were tied
  to the old wording, so this change is documentation/docstring-only.
- The remaining matches for TSDF/RGB-D/fusion terms are the new allowed-policy
  wording or this task note, not categorical bans.

## Decisions

- The policy now allows TSDF integration, RGB-D fusion, volumetric fusion, online
  mapping, and SLAM-style workflows when required by an explicit protocol or
  dataset workflow.
- Commodity fusion/integration work should be delegated to established backends
  such as Open3D and recorded with backend versions, parameters, inputs, and
  result-affecting outputs.
- Depth metrics and runners still do not hide reconstruction behavior inside
  metric computation; conversion into scene geometry must be an explicit
  pipeline/backend stage.
- No tests were added because runtime behavior did not change.

## Verification

- `rg -n --glob '!.agent/tasks/024-fusion-policy-docs.md' "fusion support is out of scope|hard boundary|never integrated into meshes|build TSDF volumes|does not integrate RGB-D|does not build a scene reconstruction|Depth integration into scene reconstructions is not part|not converted into scene reconstructions|not converted into a scene reconstruction|never turned into scene reconstructions|solely as an evaluation-time visibility-culling mechanism|no reconstruction backends|There is no TSDF|no TSDF integration|forbidden \\*\\*as reconstruction\\*\\*" CLAUDE.md .agent docs eval3r tests` — OK, no matches.
- `mkdocs build` — OK, documentation built successfully.

## Status

done
