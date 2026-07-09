# Fidelity labels

Every protocol and every result carries a **fidelity** label saying how close the
produced numbers are to a dataset's official evaluation. This matters because "we
evaluated on dataset X" can mean very different things.

## The three labels

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

### `native`

The protocol is evaluated locally by eval3r rather than by an official benchmark tool.
It may be an eval3r-defined protocol, such as the `single_*` quick-check protocols and
ScanNet community geometry protocols, or a validated local port of established dataset
semantics, such as `dtu_native_pointcloud`.

Native protocols are explicit, hashed, and repeatable, but they do **not** claim official
leaderboard comparability. For DTU, ObsMask/Plane handling is still required by the
protocol; missing files are recorded explicitly rather than ignored.

### `server`

The split's ground truth is withheld and only the official benchmark server can score
it (Tanks and Temples intermediate/advanced, ETH3D test). eval3r **refuses** to run
local metrics under a server-only protocol — a locally computed number for those splits
would be official-looking but meaningless.

## Why this is enforced

Comparing an `official` run against a `native` run is one of the comparability
warnings `e3r diff` raises ("backend officialness differs"): the numbers were produced by
different scoring machinery, so a delta between them may reflect the evaluator rather
than the method.
