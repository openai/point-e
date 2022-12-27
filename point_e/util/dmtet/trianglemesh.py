import torch

def _base_face_areas(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas

def _base_sample_points_selected_faces(face_vertices, face_features=None):
    """Base function to sample points over selected faces.
       The coordinates of the face vertices are interpolated to generate new samples.
    Args:
        face_vertices (tuple of torch.Tensor):
            Coordinates of vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
        face_features (tuple of torch.Tensor, Optional):
            Features of face vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.
    Returns:
        (torch.Tensor, torch.Tensor):
            Sampled point coordinates of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
            Sampled points interpolated features of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.
            If `face_vertices_features` arg is not specified, the returned interpolated features are None.
    """

    face_vertices0, face_vertices1, face_vertices2 = face_vertices

    sampling_shape = tuple(int(d) for d in face_vertices0.shape[:-1]) + (1,)
    # u is proximity to middle point between v1 and v2 against v0.
    # v is proximity to v2 against v1.
    #
    # The probability density for u should be f_U(u) = 2u.
    # However, torch.rand use a uniform (f_X(x) = x) distribution,
    # so using torch.sqrt we make a change of variable to have the desired density
    # f_Y(y) = f_X(y ^ 2) * |d(y ^ 2) / dy| = 2y
    u = torch.sqrt(torch.rand(sampling_shape,
                              device=face_vertices0.device,
                              dtype=face_vertices0.dtype))

    v = torch.rand(sampling_shape,
                   device=face_vertices0.device,
                   dtype=face_vertices0.dtype)
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * face_vertices0 + w1 * face_vertices1 + w2 * face_vertices2

    features = None
    if face_features is not None:
        face_features0, face_features1, face_features2 = face_features
        features = w0 * face_features0 + w1 * face_features1 + \
            w2 * face_features2

    return points, features

def sample_points(vertices, faces, num_samples, areas=None, face_features=None):
    r"""Uniformly sample points over the surface of triangle meshes.
    First face on which the point is sampled is randomly selected,
    with the probability of selection being proportional to the area of the face.
    then the coordinate on the face is uniformly sampled.
    If ``face_features`` is defined for the mesh faces,
    the sampled points will be returned with interpolated features as well,
    otherwise, no feature interpolation will occur.
    Args:
        vertices (torch.Tensor):
            The vertices of the meshes, of shape
            :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            The faces of the mesh, of shape :math:`(\text{num_faces}, 3)`.
        num_samples (int):
            The number of point sampled per mesh.
        areas (torch.Tensor, optional):
            The areas of each face, of shape :math:`(\text{batch_size}, \text{num_faces})`,
            can be preprocessed, for fast on-the-fly sampling,
            will be computed if None (default).
        face_features (torch.Tensor, optional):
            Per-vertex-per-face features, matching ``faces`` order,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`.
            For example:
                1. Texture uv coordinates would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`.
                2. RGB color values would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.
            When specified, it is used to interpolate the features for new sampled points.
    See also:
        :func:`~kaolin.ops.mesh.index_vertices_by_faces` for conversion of features defined per vertex
        and need to be converted to per-vertex-per-face shape of :math:`(\text{num_faces}, 3)`.
    Returns:
        (torch.Tensor, torch.LongTensor, (optional) torch.Tensor):
            the pointclouds of shape :math:`(\text{batch_size}, \text{num_samples}, 3)`,
            and the indexes of the faces selected,
            of shape :math:`(\text{batch_size}, \text{num_samples})`.
            If ``face_features`` arg is specified, then the interpolated features of sampled points of shape
            :math:`(\text{batch_size}, \text{num_samples}, \text{feature_dim})` are also returned.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("sample_points is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)        # (num_faces, 3) -> tuple of (num_faces,)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))  # (batch_size, num_faces, 3)

    if areas is None:
        areas = _base_face_areas(face_v_0, face_v_1, face_v_2).squeeze(-1)
    face_dist = torch.distributions.Categorical(areas)
    face_choices = face_dist.sample([num_samples]).transpose(0, 1)
    _face_choices = face_choices.unsqueeze(-1).repeat(1, 1, 3)
    v0 = torch.gather(face_v_0, 1, _face_choices)  # (batch_size, num_samples, 3)
    v1 = torch.gather(face_v_1, 1, _face_choices)  # (batch_size, num_samples, 3)
    v2 = torch.gather(face_v_2, 1, _face_choices)  # (batch_size, num_samples, 3)
    face_vertices_choices = (v0, v1, v2)

    # UV coordinates are available, make sure to calculate them for sampled points as well
    face_features_choices = None
    if face_features is not None:
        feat_dim = face_features.shape[-1]
        # (num_faces, 3) -> tuple of (num_faces,)
        _face_choices = face_choices[..., None, None].repeat(1, 1, 3, feat_dim)
        face_features_choices = torch.gather(face_features, 1, _face_choices)
        face_features_choices = tuple(
            tmp_feat.squeeze(2) for tmp_feat in torch.split(face_features_choices, 1, dim=2))

    points, point_features = _base_sample_points_selected_faces(
        face_vertices_choices, face_features_choices)

    if point_features is not None:
        return points, face_choices, point_features
    else:
        return points, face_choices