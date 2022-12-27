from typing import TextIO, Optional

import numpy as np


def write_txt(
        f: TextIO,
        coords: np.ndarray,
        rgb: Optional[np.ndarray] = None,
):
    """
    Write a text file for a point cloud.

    :param f: an i/o stream to write to, such as the stream returned by open()
    :param coords: an [N x 3] array of floating point coordinates.
    :param rgb: an [N x 3] array of vertex colors, in the range [0.0, 1.0].

    """
    points = coords
    if rgb is not None:
        rgb = (rgb * 255.499).round().astype(int)
        points = np.concatenate([points, rgb], axis=1)
    np.savetxt(f, points)
