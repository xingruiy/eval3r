# Fidelity labels

Every protocol and every result carries a **fidelity** label saying how close the
produced numbers are to a dataset's official evaluation. This matters because "we
evaluated on dataset X" can mean very different things.

## The four labels

### `official`

The protocol delegates scoring to the **real official script or benchmark tool**, run as
a subprocess, and records the official backend's version/commit and command. eval3r
never reimplements, stubs, or approximates an official evaluator — if the real tool
cannot run, the run fails explicitly rather than producing lookalike numbers.

Built-ins: `tanks_temples_training_official` (official python toolbox, unmodified, under
its pinned `open3d==0.9` interpreter), `eth3d_training_official` (official
multi-view-evaluation binary — its voxel-normalized scoring with beam-based free-space
classification is not reproducible with plain distance metrics).

When an official tool needs a *mechanical* compatibility fix to run at all (a renamed
import, a build flag), the exact patch is recorded in result metadata. Anything that
could change the numbers is never patched — eval3r stops and reports instead.

### `official_like`

The protocol reimplements established dataset semantics locally as a **validated port**,
regression-tested against the reference implementation. Built-in:
`dtu_official_like_pointcloud` — a Python port of the official DTU MATLAB evaluation
(ObsMask + Plane handling included), validated against the reference code on real data.
Skipping ObsMask/Plane would downgrade this fidelity; eval3r records missing Plane files
explicitly instead of ignoring them.

### `eval3r_native`

The protocol is defined by eval3r. It is explicit, hashed, and repeatable, but it does
**not** claim official leaderboard comparability. Built-ins: the `single_*` protocols
and the ScanNet geometry protocols (ScanNet has no official reconstruction benchmark;
they follow the community 5 cm F-score convention).

### `server_only`

The split's ground truth is withheld and only the official benchmark server can score
it (Tanks and Temples intermediate/advanced, ETH3D test). eval3r **refuses** to run
local metrics under a server-only protocol — a locally computed number for those splits
would be official-looking but meaningless.

## Why this is enforced

Comparing an `official` run against an `eval3r_native` run is one of the comparability
warnings `e3r diff` raises ("backend officialness differs"): the numbers were produced by
different scoring machinery, so a delta between them may reflect the evaluator rather
than the method.
