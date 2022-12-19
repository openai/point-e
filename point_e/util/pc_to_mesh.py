from typing import Dict

import numpy as np
import skimage
import torch
from tqdm.auto import tqdm

from point_e.models.sdf import PointCloudSDFModel

from .mesh import TriMesh
from .point_cloud import PointCloud


def marching_cubes_mesh(
    pc: PointCloud,
    model: PointCloudSDFModel,
    batch_size: int = 4096,
    grid_size: int = 128,
    side_length: float = 1.02,
    fill_vertex_channels: bool = True,
    progress: bool = False,
) -> TriMesh:
    """
    Run marching cubes on the SDF predicted from a point cloud to produce a
    mesh representing the 3D surface.

    :param pc: the point cloud to apply marching cubes to.
    :param model: the model to use to predict SDF values.
    :param grid_size: the number of samples along each axis. A total of
                      grid_size**3 function evaluations are performed.
    :param side_length: the size of the cube containing the model, which is
                        assumed to be centered at the origin.
    :param fill_vertex_channels: if True, use the nearest neighbor of each mesh
                                 vertex in the point cloud to compute vertex
                                 data (e.g. colors).
    """
    voxel_size = side_length / (grid_size - 1)
    min_coord = -side_length / 2

    def int_coord_to_float(int_coords: torch.Tensor) -> torch.Tensor:
        return int_coords.float() * voxel_size + min_coord

    with torch.no_grad():
        cond = model.encode_point_clouds(
            torch.from_numpy(pc.coords).permute(1, 0).to(model.device)[None]
        )

    indices = range(0, grid_size**3, batch_size)
    if progress:
        indices = tqdm(indices)

    volume = []
    for i in indices:
        indices = torch.arange(
            i, min(i + batch_size, grid_size**3), step=1, dtype=torch.int64, device=model.device
        )
        zs = int_coord_to_float(indices % grid_size)
        ys = int_coord_to_float(torch.div(indices, grid_size, rounding_mode="trunc") % grid_size)
        xs = int_coord_to_float(torch.div(indices, grid_size**2, rounding_mode="trunc"))
        coords = torch.stack([xs, ys, zs], dim=0)
        with torch.no_grad():
            volume.append(model(coords[None], encoded=cond)[0])
    volume_np = torch.cat(volume).view(grid_size, grid_size, grid_size).cpu().numpy()

    if np.all(volume_np < 0) or np.all(volume_np > 0):
        # The volume is invalid for some reason, which will break
        # marching cubes unless we center it.
        volume_np -= np.mean(volume_np)

    verts, faces, normals, _ = skimage.measure.marching_cubes(
        volume=volume_np,
        level=0,
        allow_degenerate=False,
        spacing=(voxel_size,) * 3,
    )

    # The triangles follow the left-hand rule, but we want to
    # follow the right-hand rule.
    # This syntax might seem roundabout, but we get incorrect
    # results if we do: x[:,0], x[:,1] = x[:,1], x[:,0]
    old_f1 = faces[:, 0].copy()
    faces[:, 0] = faces[:, 1]
    faces[:, 1] = old_f1

    verts += min_coord
    return TriMesh(
        verts=verts,
        faces=faces,
        normals=normals,
        vertex_channels=None if not fill_vertex_channels else _nearest_vertex_channels(pc, verts),
    )


def _nearest_vertex_channels(pc: PointCloud, verts: np.ndarray) -> Dict[str, np.ndarray]:
    nearest = pc.nearest_points(verts)
    return {ch: arr[nearest] for ch, arr in pc.channels.items()}
