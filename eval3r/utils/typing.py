"""Type aliases used across eval3r."""

from __future__ import annotations

import os
from typing import Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, os.PathLike[str]]

# Convenience aliases — all are float arrays unless otherwise documented.
Points = NDArray[np.floating]  # (N, 3)
Faces = NDArray[np.integer]  # (M, 3)
Colors = NDArray[np.unsignedinteger]  # (N, 3) uint8
Pose = NDArray[np.floating]  # (4, 4)
Poses = NDArray[np.floating]  # (T, 4, 4)
