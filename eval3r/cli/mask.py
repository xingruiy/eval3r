"""``e3r mask`` — generate, inspect, and visualize volumetric occlusion masks.

A single ``e3r mask gen`` command builds masks via TSDF-style carving. It
accepts two input shapes:

- ``--depth-path <dir> --depth-pattern '{frame:06d}.png' …`` — manual sensor
  depth mode.
- ``--mesh-path <mesh.ply> --image-size 640x480 …`` — manual rendered-depth
  mode (requires the ``[render]`` extra).

Pattern syntax is Python ``str.format`` with a ``{frame}`` field.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import typer

from eval3r.filtering.occlusion.generate import from_depth, from_rendered
from eval3r.filtering.occlusion.mask import (
    OcclusionFilter,
    load_occlusion_mask,
    save_occlusion_mask,
)
from eval3r.io.geometry import load_mesh
from eval3r.io.trajectory import Trajectory, load_trajectory_auto

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_image_size(spec: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[x,X]\s*(\d+)\s*", spec)
    if not m:
        raise typer.BadParameter(
            f"--image-size must be WxH (e.g. 640x480), got {spec!r}"
        )
    return int(m.group(1)), int(m.group(2))


def _resolve_per_frame_paths(
    path: Path,
    pattern: str | None,
    *,
    asset_label: str,
    default_glob: str,
) -> list[Path]:
    """Enumerate per-frame files from ``path`` according to ``pattern``."""
    if not path.exists():
        raise typer.BadParameter(f"--{asset_label}-path does not exist: {path}")
    if path.is_file():
        raise typer.BadParameter(
            f"--{asset_label}-path must be a directory when discovering per-frame "
            f"files; got file {path}. Pass a directory plus --{asset_label}-pattern."
        )
    if pattern is not None:
        files: list[Path] = []
        i = 0
        consecutive_misses = 0
        while consecutive_misses < 1:
            candidate = path / pattern.format(frame=i)
            if candidate.exists():
                files.append(candidate)
                consecutive_misses = 0
            else:
                consecutive_misses += 1
            i += 1
            if i > 1_000_000:  # pragma: no cover - safety bound
                break
        if not files:
            raise typer.BadParameter(
                f"No {asset_label} files matched pattern {pattern!r} under {path}"
            )
        return files

    matches = sorted(path.glob(default_glob), key=_natural_key)
    if not matches:
        raise typer.BadParameter(
            f"No {asset_label} files matched glob {default_glob!r} under {path}; "
            f"pass --{asset_label}-pattern to be explicit."
        )
    return matches


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(p: Path) -> tuple:
    parts = _NUM_RE.split(p.stem)
    return tuple(int(s) if s.isdigit() else s for s in parts)


def _load_depth_image(path: Path) -> np.ndarray:
    from eval3r.utils.optional import optional_import

    imageio = optional_import("imageio.v3", extra="render")
    return np.asarray(imageio.imread(path))


def _load_pose_matrix(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape == (4, 4):
        return arr.astype(np.float64)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    raise typer.BadParameter(
        f"Pose file {path} must contain a 3×4 or 4×4 matrix; got shape {arr.shape}"
    )


def _load_intrinsics(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape == (3, 3):
        return arr.astype(np.float64)
    if arr.shape == (4, 4):
        return arr[:3, :3].astype(np.float64)
    raise typer.BadParameter(
        f"--intrinsics-path {path} must contain a 3×3 (or 4×4) matrix; got {arr.shape}"
    )


def _load_poses_any(
    poses_path: Path,
    pattern: str | None,
) -> tuple[np.ndarray, str]:
    """Return ``(poses (T,4,4), convention)``."""
    if poses_path.is_file() and pattern is None:
        traj: Trajectory = load_trajectory_auto(poses_path)
        return traj.poses, traj.convention
    files = _resolve_per_frame_paths(
        poses_path, pattern, asset_label="poses", default_glob="*.txt"
    )
    poses = np.stack([_load_pose_matrix(f) for f in files], axis=0)
    return poses, "unspecified"


# ---------------------------------------------------------------------------
# gen — single command, branches on inputs
# ---------------------------------------------------------------------------


@app.command("gen")
def gen_cmd(
    out_dir: str = typer.Option(..., "--out-dir", help="Output directory."),
    # --- manual depth mode ---
    depth_path: str | None = typer.Option(
        None, "--depth-path",
        help="Manual depth mode: directory containing per-frame depth images.",
    ),
    depth_pattern: str | None = typer.Option(
        None, "--depth-pattern",
        help="Filename pattern under --depth-path (e.g. '{frame:06d}.png').",
    ),
    # --- manual rendered mode ---
    mesh_path: str | None = typer.Option(
        None, "--mesh-path",
        help="Manual rendered mode: GT mesh file. Requires --image-size.",
    ),
    image_size: str = typer.Option(
        "640x480", "--image-size", help="Render resolution as WxH (rendered mode)."
    ),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
    # --- shared manual flags ---
    poses_path: str | None = typer.Option(
        None, "--poses-path",
        help="Manual mode: trajectory file or directory of per-frame pose files.",
    ),
    poses_pattern: str | None = typer.Option(
        None, "--poses-pattern",
        help="Pose filename pattern when --poses-path is a directory.",
    ),
    intrinsics_path: str | None = typer.Option(
        None, "--intrinsics-path",
        help="Manual mode: 3×3 (or 4×4) intrinsics text file.",
    ),
    # --- frame conventions ---
    pose_convention: str = typer.Option("T_cw", "--pose-convention"),
    camera_frame: str = typer.Option("opencv", "--camera-frame"),
    # --- depth handling ---
    depth_scale: float = typer.Option(
        1.0, "--depth-scale",
        help="Divide raw depth by this value (e.g. 1000 for mm→m).",
    ),
    # --- carving knobs ---
    voxel_size: float = typer.Option(0.02, "--voxel-size"),
    margin: float = typer.Option(0.1, "--margin"),
    max_depth: float = typer.Option(
        3.5, "--max-depth",
        help=(
            "Maximum depth in metres. Discards observed depth pixels farther "
            "than this and bounds carving/projection. Default: 3.5."
        ),
    ),
    near: float = typer.Option(
        0.1, "--near", help="Frustum near plane (metres)."
    ),
    truncation: float | None = typer.Option(
        None, "--truncation",
        help="TSDF band past surface (metres). Defaults to 4 * voxel_size.",
    ),
    dilation: int = typer.Option(0, "--dilation"),
    # --- frame selection ---
    frames: str | None = typer.Option(
        None, "--frames", help="Comma-separated frame ids, e.g. '0,5,10'."
    ),
    frames_file: str | None = typer.Option(
        None, "--frames-file", help="Path to a text file with one frame id per line."
    ),
    frame_stride: int = typer.Option(
        5, "--frame-stride",
        help="Process every Nth frame (default 5).",
    ),
    max_frames: int | None = typer.Option(None, "--max-frames"),
) -> None:
    """Generate a volumetric occlusion mask via TSDF carving.

    Branches on inputs:

    - ``--depth-path …`` → manual depth carving.
    - ``--mesh-path …`` → manual rendered carving (requires ``--image-size``).
    """
    if pose_convention not in ("T_cw", "T_wc"):
        raise typer.BadParameter("--pose-convention must be T_cw or T_wc")
    if camera_frame not in ("opencv", "opengl"):
        raise typer.BadParameter("--camera-frame must be opencv or opengl")
    if depth_path is not None and mesh_path is not None:
        raise typer.BadParameter(
            "--depth-path and --mesh-path are mutually exclusive."
        )

    if depth_path is not None:
        mask = _gen_from_depth_paths(
            depth_path=depth_path,
            depth_pattern=depth_pattern,
            poses_path=poses_path,
            poses_pattern=poses_pattern,
            intrinsics_path=intrinsics_path,
            voxel_size=voxel_size,
            margin=margin,
            pose_convention=pose_convention,
            camera_frame=camera_frame,
            depth_scale=depth_scale,
            max_depth=max_depth,
            near=near,
            truncation=truncation,
            dilation=dilation,
            frames=frames,
            frames_file=frames_file,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
    elif mesh_path is not None:
        mask = _gen_from_mesh_paths(
            mesh_path=mesh_path,
            poses_path=poses_path,
            poses_pattern=poses_pattern,
            intrinsics_path=intrinsics_path,
            image_size=image_size,
            voxel_size=voxel_size,
            margin=margin,
            pose_convention=pose_convention,
            camera_frame=camera_frame,
            max_depth=max_depth,
            near=near,
            truncation=truncation,
            dilation=dilation,
            frames=frames,
            frames_file=frames_file,
            frame_stride=frame_stride,
            max_frames=max_frames,
            headless=headless,
        )
    else:
        raise typer.BadParameter(
            "Specify one of:\n"
            "  --depth-path <dir> --poses-path <…> --intrinsics-path <K.txt>\n"
            "  --mesh-path <ply> --poses-path <…> --intrinsics-path <K.txt> --image-size WxH"
        )

    mp, tp = save_occlusion_mask(mask, out_dir)
    typer.echo(f"wrote {mp}")
    typer.echo(f"wrote {tp}")



def _gen_from_depth_paths(
    *,
    depth_path: str,
    depth_pattern: str | None,
    poses_path: str | None,
    poses_pattern: str | None,
    intrinsics_path: str | None,
    voxel_size: float,
    margin: float,
    pose_convention: str,
    camera_frame: str,
    depth_scale: float,
    max_depth: float,
    near: float,
    truncation: float | None,
    dilation: int,
    frames: str | None,
    frames_file: str | None,
    frame_stride: int,
    max_frames: int | None,
) -> OcclusionFilter:
    if poses_path is None or intrinsics_path is None:
        raise typer.BadParameter(
            "Manual depth mode needs --depth-path, --poses-path, and --intrinsics-path."
        )
    depth_files = _resolve_per_frame_paths(
        Path(depth_path), depth_pattern, asset_label="depth", default_glob="*.png"
    )
    poses, used_convention = _load_poses_any(Path(poses_path), poses_pattern)
    if len(poses) != len(depth_files):
        raise typer.BadParameter(
            f"Got {len(depth_files)} depth files but {len(poses)} poses; counts must match."
        )
    K = _load_intrinsics(Path(intrinsics_path))
    depth_maps = [_load_depth_image(p) for p in depth_files]
    convention = (
        used_convention
        if used_convention in ("T_cw", "T_wc")
        else pose_convention
    )
    try:
        return from_depth(
            depth_maps,
            poses,
            K,
            voxel_size=voxel_size,
            margin=margin,
            pose_convention=convention,  # type: ignore[arg-type]
            camera_frame=camera_frame,  # type: ignore[arg-type]
            depth_scale=depth_scale,
            depth_max=max_depth,
            max_depth=max_depth,
            near=near,
            truncation=truncation,
            frames=frames,
            frames_file=frames_file,
            frame_stride=frame_stride,
            max_frames=max_frames,
            dilation=dilation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _gen_from_mesh_paths(
    *,
    mesh_path: str,
    poses_path: str | None,
    poses_pattern: str | None,
    intrinsics_path: str | None,
    image_size: str,
    voxel_size: float,
    margin: float,
    pose_convention: str,
    camera_frame: str,
    max_depth: float,
    near: float,
    truncation: float | None,
    dilation: int,
    frames: str | None,
    frames_file: str | None,
    frame_stride: int,
    max_frames: int | None,
    headless: bool,
) -> OcclusionFilter:
    if poses_path is None or intrinsics_path is None:
        raise typer.BadParameter(
            "Manual rendered mode needs --mesh-path, --poses-path, and --intrinsics-path."
        )
    geom = load_mesh(mesh_path)
    poses, used_convention = _load_poses_any(Path(poses_path), poses_pattern)
    K = _load_intrinsics(Path(intrinsics_path))
    convention = (
        used_convention
        if used_convention in ("T_cw", "T_wc")
        else pose_convention
    )
    return from_rendered(
        geom,
        poses,
        K,
        _parse_image_size(image_size),
        voxel_size=voxel_size,
        margin=margin,
        pose_convention=convention,  # type: ignore[arg-type]
        camera_frame=camera_frame,  # type: ignore[arg-type]
        max_depth=max_depth,
        near=near,
        truncation=truncation,
        frames=frames,
        frames_file=frames_file,
        frame_stride=frame_stride,
        max_frames=max_frames,
        dilation=dilation,
        headless=headless,
    )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command("inspect")
def inspect_cmd(
    mask: str = typer.Option(..., "--mask", help="Path to occlusion_mask.npy."),
    t_mask_scene: str = typer.Option(
        ..., "--t-mask-scene", help="Path to T_mask_scene.txt."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Print mask metadata (shape, voxel size, world bbox, visible fraction)."""
    m: OcclusionFilter = load_occlusion_mask(mask, t_mask_scene)
    grid = m.grid
    T = m.T_mask_scene

    T_inv = np.linalg.inv(T)
    dims = np.array(grid.shape, dtype=np.float64)
    corners_idx = np.array(
        [
            [0, 0, 0, 1],
            [dims[0] - 1, 0, 0, 1],
            [0, dims[1] - 1, 0, 1],
            [0, 0, dims[2] - 1, 1],
            [dims[0] - 1, dims[1] - 1, dims[2] - 1, 1],
        ],
        dtype=np.float64,
    )
    corners_world = (T_inv @ corners_idx.T).T[:, :3]
    bbox_min = corners_world.min(axis=0)
    bbox_max = corners_world.max(axis=0)

    diag = np.diag(T)[:3]
    voxel_size = float(np.mean(1.0 / np.where(np.abs(diag) > 0, diag, 1.0)))

    visible = int((grid < 0.5).sum())
    total = int(grid.size)

    payload = {
        "source": m.source,
        "shape": list(grid.shape),
        "voxel_size": voxel_size,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "visible_voxels": visible,
        "total_voxels": total,
        "visible_fraction": visible / total if total else 0.0,
    }
    if json_out:
        import json

        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"source           : {payload['source']}")
    typer.echo(f"shape            : {payload['shape']}")
    typer.echo(f"voxel_size       : {payload['voxel_size']:.6f}")
    typer.echo(
        f"bbox_min         : "
        f"[{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]"
    )
    typer.echo(
        f"bbox_max         : "
        f"[{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]"
    )
    typer.echo(
        f"visible / total  : {visible} / {total} "
        f"({payload['visible_fraction']:.4f})"
    )


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------


@app.command("visualize")
def visualize_cmd(
    mask: str = typer.Option(..., "--mask", help="Path to occlusion_mask.npy."),
    t_mask_scene: str | None = typer.Option(
        None,
        "--t-mask-scene",
        help="Optional T_mask_scene.txt. Defaults to a sibling T_mask_scene.txt when present.",
    ),
    out_dir: str = typer.Option("mask_viz", "--out-dir", help="Output directory."),
    value: str = typer.Option(
        "occluded",
        "--value",
        help="Voxel class to visualize: occluded (grid >= 0.5) or visible (grid < 0.5).",
    ),
    max_dim: int = typer.Option(
        160,
        "--max-dim",
        help="Maximum downsampled grid dimension for the PLY point cloud.",
    ),
    threshold: float = typer.Option(
        0.25,
        "--threshold",
        help="Block occupancy fraction needed to keep a downsampled cell.",
    ),
    max_points: int = typer.Option(
        500_000,
        "--max-points",
        help="Maximum surface points written to the PLY file.",
    ),
) -> None:
    """Export an overview PNG and an orbitable downsampled PLY point cloud."""
    if value not in {"occluded", "visible"}:
        raise typer.BadParameter("--value must be 'occluded' or 'visible'")

    from eval3r.filtering.occlusion.visualize import save_mask_overview, save_mask_point_cloud

    mask_path = Path(mask)
    grid = np.load(mask_path, mmap_mode="r")
    if grid.ndim != 3:
        raise typer.BadParameter(f"--mask must be a 3D .npy array, got shape {grid.shape}")

    t_path = Path(t_mask_scene) if t_mask_scene is not None else mask_path.with_name("T_mask_scene.txt")
    T = np.loadtxt(t_path) if t_path.exists() else None

    out = Path(out_dir)
    overview_path = save_mask_overview(grid, out / f"{value}_overview.png", value=value)  # type: ignore[arg-type]
    ply_path, n_points, factors = save_mask_point_cloud(
        grid,
        out / f"{value}_surface.ply",
        value=value,  # type: ignore[arg-type]
        max_dim=max_dim,
        threshold=threshold,
        max_points=max_points,
        t_mask_scene=T,
    )

    coord_space = "scene/world" if T is not None else "voxel"
    typer.echo(f"overview_png     : {overview_path}")
    typer.echo(f"surface_ply      : {ply_path}")
    typer.echo(f"value            : {value}")
    typer.echo(f"shape            : {list(grid.shape)}")
    typer.echo(f"downsample       : {list(factors)}")
    typer.echo(f"surface_points   : {n_points}")
    typer.echo(f"coordinate_space : {coord_space}")
