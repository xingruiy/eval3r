from eval3r.presets import (
    dtu,
    eth3d,
    replica,
    scannet,
    tanks_temples,
    tum_rgbd,
)


# Back-compat alias exported for star-import users.
tnt = tanks_temples

PRESETS = {
    "scannet": scannet.SCANNET_PRESET,
    "tum_rgbd": tum_rgbd.TUM_RGBD_PRESET,
    "replica": replica.REPLICA_PRESET,
    "dtu": dtu.DTU_PRESET,
    "eth3d": eth3d.ETH3D_PRESET,
    "tanks_temples": tanks_temples.TANKS_TEMPLES_PRESET,
    "tnt": tanks_temples.TANKS_TEMPLES_PRESET,
}

__all__ = [
    "PRESETS",
    "dtu",
    "eth3d",
    "replica",
    "scannet",
    "tanks_temples",
    "tnt",
    "tum_rgbd",
]
