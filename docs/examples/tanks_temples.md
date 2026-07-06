# Example: Tanks and Temples

Tanks and Temples training-split GT is an **independent laser scan** with public
per-scene evaluation assets; scoring is done by the **real official toolbox** (fidelity
`official`). The intermediate and advanced splits are **server-only** — GT is withheld,
and eval3r refuses to evaluate them locally.

## One-time setup: the official toolbox

The official toolbox is a user-supplied checkout run under its pinned `open3d==0.9`
interpreter — see [Installation](../install.md#tanks-and-temples-official-toolbox):

```bash
export EVAL3R_TNT_TOOLBOX=/path/to/TanksAndTemples/python_toolbox/evaluation
export EVAL3R_TNT_PYTHON=/path/to/envs/tnt_toolbox/bin/python
```

## Dataset layout

One directory per training scene, holding the five official evaluation assets:

```text
<root>/Barn/Barn.ply                        laser-scan GT point cloud (metres)
<root>/Barn/Barn.json                       crop volume
<root>/Barn/Barn_trans.txt                  4x4 alignment transform
<root>/Barn/Barn_COLMAP_SfM.log             reference SfM trajectory (.log)
<root>/Barn/Barn_mapping_reference.txt      image → trajectory mapping
```

Scenes are discovered from the official training list (directories present under the
root); an optional `<root>/training.txt` restricts evaluation to a subset.

## Predictions

One point cloud per scene, named by scene:

```text
preds/Barn.ply
preds/Truck.ply
```

## Run

```bash
e3r benchmark run preds/ --dataset tanks_temples --split training \
  --protocol tanks_temples_training_official --root /data/tnt \
  --method mymethod --out runs/tnt_mymethod
```

## What the official toolbox owns

ICP refinement, crop-volume application, and the **per-scene** distance threshold
(`dTau`) all come from the official toolbox — eval3r reimplements none of it, and the
per-scene threshold is read from the official output, never hardcoded. Reported metrics
are the official `precision` / `recall` / `fscore` at each scene's own threshold. The
resolved toolbox directory, its git commit, the interpreter used, and the exact command
are recorded in result metadata.

Attempting the withheld splits fails in preflight:

```bash
e3r benchmark run preds/ --dataset tanks_temples --split intermediate \
  --protocol tanks_temples_intermediate_server_only --root /data/tnt
# refused: GT for this split is withheld; submit to the official benchmark server.
```
