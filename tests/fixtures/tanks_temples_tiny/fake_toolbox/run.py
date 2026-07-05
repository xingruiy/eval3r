"""Fake stand-in for the official Tanks and Temples ``run.py`` (test fixture only).

This is **not** the official evaluator. It mimics the official toolbox's command-line
interface and the exact format of its printed summary so eval3r's ``tnt_official``
wrapper can be exercised end-to-end (argument construction, subprocess invocation,
output parsing, artifact resolution) without the heavy open3d-based real toolbox.

It validates that the wrapper passed a scene directory containing the expected GT/crop/
trans artifacts and a readable prediction ply, then prints fixed precision/recall/
F-score and the per-scene distance tau in the official ``label : value`` format.
"""

import argparse
import os
import sys

# Per-scene distance thresholds, exactly as the official config.py scenes_tau_dict.
SCENES_TAU = {
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--traj-path", required=True)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    scene = os.path.basename(os.path.normpath(args.dataset_dir))
    if scene not in SCENES_TAU:
        sys.stderr.write(f"unknown scene {scene}\n")
        return 2

    required = [
        os.path.join(args.dataset_dir, scene + ".ply"),
        os.path.join(args.dataset_dir, scene + ".json"),
        os.path.join(args.dataset_dir, scene + "_trans.txt"),
        args.traj_path,
        args.ply_path,
    ]
    for path in required:
        if not os.path.isfile(path):
            sys.stderr.write(f"missing required input: {path}\n")
            return 3

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, scene + ".evaluation.txt"), "w") as fh:
        fh.write("fake evaluation output\n")

    dtau = SCENES_TAU[scene]
    print("===========================")
    print(f"evaluation result : {scene}")
    print("===========================")
    print(f"distance tau : {dtau:.3f}")
    print(f"precision : {0.8500:.4f}")
    print(f"recall : {0.7500:.4f}")
    print(f"f-score : {0.7969:.4f}")
    print("===========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
