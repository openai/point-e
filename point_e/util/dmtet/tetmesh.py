import torch

__all__ = ['marching_tetrahedra']

triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


def _sort_edges(edges):
    """sort last dimension of edges of shape (E, 2)"""
    with torch.no_grad():
        order = (edges[:, 0] > edges[:, 1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges, index=order, dim=1)
        b = torch.gather(input=edges, index=1 - order, dim=1)

    return torch.stack([a, b], -1)


def _unbatched_marching_tetrahedra(vertices, tets, sdf, return_tet_idx):
    """unbatched marching tetrahedra.
    Refer to :func:`marching_tetrahedra`.
    """
    device = vertices.device
    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]
    edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    edges_to_interp_sdf[:, -1] *= -1

    denominator = edges_to_interp_sdf.sum(1, keepdim=True)

    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    if return_tet_idx:
        tet_idx = torch.arange(tets.shape[0], device=device)[valid_tets]
        tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles ==
                            2].unsqueeze(-1).expand(-1, 2).reshape(-1)), dim=0)
        return verts, faces, tet_idx
    return verts, faces


def marching_tetrahedra(vertices, tets, sdf, return_tet_idx=False):
    r"""Convert discrete signed distance fields encoded on tetrahedral grids to triangle 
    meshes using marching tetrahedra algorithm as described in `An efficient method of 
    triangulating equi-valued surfaces by using tetrahedral cells`_. The output surface is differentiable with respect to
    input vertex positions and the SDF values. For more details and example usage in learning, see 
    `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.
    Args:
        vertices (torch.tensor): batched vertices of tetrahedral meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        tets (torch.tensor): unbatched tetrahedral mesh topology, of shape
                             :math:`(\text{num_tetrahedrons}, 4)`.
        sdf (torch.tensor): batched SDFs which specify the SDF value of each vertex, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.
        return_tet_idx (optional, bool): if True, return index of tetrahedron
                                         where each face is extracted. Default: False.
    Returns:
        (list[torch.Tensor], list[torch.LongTensor], (optional) list[torch.LongTensor]): 
            - the list of vertices for mesh converted from each tetrahedral grid.
            - the list of faces for mesh converted from each tetrahedral grid.
            - the list of indices that correspond to tetrahedra where faces are extracted.
    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]]], dtype=torch.float)
        >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([[-1., -1., 0.5, 0.5]], dtype=torch.float)
        >>> verts_list, faces_list, tet_idx_list = marching_tetrahedra(vertices, tets, sdf, True)
        >>> verts_list[0]
        tensor([[0.0000, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.6667],
                [0.3333, 0.6667, 0.0000],
                [0.3333, 0.0000, 0.6667]])
        >>> faces_list[0]
        tensor([[3, 0, 1],
                [3, 2, 0]])
        >>> tet_idx_list[0]
        tensor([0, 0])
    .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
        https://search.ieice.org/bin/summary.php?id=e74-d_1_214
    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """

    list_of_outputs = [_unbatched_marching_tetrahedra(vertices[b], tets, sdf[b], return_tet_idx) for b in range(vertices.shape[0])]
    return list(zip(*list_of_outputs))