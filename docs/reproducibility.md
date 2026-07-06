# Reproducibility

A result is useful only when it can be understood, audited, and rerun. eval3r records the
protocol, dataset variant, ground-truth meaning, backend choices, failures, and
environment for every run. No number is reported without enough metadata to know what it
means.

Principles:

```text
A metric number is inseparable from its protocol (and its hash).
A dataset name is not enough; variant and GT provenance are recorded.
Partial scene coverage is always visible.
Backend choices and versions are always recorded.
Official-like claims require regression tests or wrapped official behavior.
Server-only splits never produce local official-looking numbers.
```

## What every run records

See [Result schema](schema.md) for the run directory and `results.json` contents. Beyond
metrics: the protocol name/version/**hash**, fidelity, dataset variant, GT
provenance/independence/density and **fingerprint**, local-evaluation status, the
alignment/masking/sampling/confidence/failure policies in effect, recorded overrides,
scene coverage and structured failures, backend names and versions (including official
tool commits and any compatibility patch applied to make an official tool run), the
resolved command, and platform/Python facts.

## Ground-truth fingerprinting

Fingerprints identify the reference actually used, not just its dataset name: DTU hashes
the GT point cloud (ObsMask/Plane availability is recorded per scene); Tanks and Temples
jointly hashes GT point cloud + crop volume + alignment transform; ETH3D jointly hashes
`scan_alignment.mlp` + every referenced scan PLY; ScanNet hashes the GT mesh variant the
protocol selected.

## Environment metadata privacy

`environment.json` deliberately contains only minimal, non-identifying facts: eval3r
version, Python version/implementation, platform/OS/CPU architecture, and the resolved
command. It never contains secrets, environment variables, working directories or other
absolute local paths, hostnames, usernames, git state, or timestamps. (A single ordering
`timestamp` lives on the RunResult itself, and `git_commit` is an optional field you can
set explicitly.)

## Report comparability

Reports and `e3r diff` make incomparability visible. A warning fires when any of these
differ between two runs:

```text
protocol hash          (loose mode only — strict mode refuses outright)
GT provenance / independence
local evaluation status
scene coverage
failure policy
alignment mode
confidence policy
sampling counts
backend officialness   (fidelity and the official_eval backend used)
```

`e3r diff` refuses strict comparisons when protocol hashes differ; `--loose` allows the
comparison but labels every output NON-STRICT.

## Determinism

Sampling seeds are protocol-pinned (derived per scene from the protocol base seed and
scene ID), so repeated runs under the same protocol hash reproduce the same numbers.
Overrides are recorded and change the hash, so a tweaked run can never masquerade as the
canonical protocol.

## Data and credentials

Run directories and committed fixtures never contain API keys, tokens, cookies,
benchmark-server credentials, licensed datasets, or large raw data. CI uses tiny
synthetic fixtures; instructions for acquiring full datasets live in the docs, not the
repository.
