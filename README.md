# eval3r Final Planning Documents

This folder contains the final planning documents for `eval3r`.

Files:

```text
.agent/plan.md              main project plan
.agent/schema.md            core schema and result model
.agent/protocols.md         protocol rules and built-in protocol templates
.agent/datasets.md          dataset adapter responsibilities and dataset-specific notes
.agent/metrics.md           metric definitions, aggregation, and tests
.agent/backends.md          backend delegation policy and optional dependencies
.agent/reproducibility.md   result files, hashing, fingerprints, and run metadata
CLAUDE.md                 rules and guides for coding agents
```

The plan intentionally excludes TSDF integration, RGB-D fusion, volumetric fusion, and online mapping. Depth sequences are supported for depth metrics only.

`.agent/datasets.md` also records adapter capability semantics, local-evaluation status, dataset variants, and assumptions that must be verified before freezing dataset fixtures.
