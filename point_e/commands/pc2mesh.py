from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.dmtet.dmtet_network import Decoder
from point_e.util.dmtet.trianglemesh import sample_points
from point_e.util.dmtet.pointcloud import chamfer_distance
from point_e.util.dmtet.tetmesh import marching_tetrahedra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

methods = ['dmtet', 'marching_cubes']
def pc_to_mesh(pc, method='dmtet', grid_res=128, lr=1e-3,
                laplacian_weight=0.15, iterations=5000,
                save_every=500,
                multires=4,
               ):
    if method == 'marching_cubes':
        #print('creating SDF model...')
        name = 'sdf'
        model = model_from_config(MODEL_CONFIGS[name], device)
        model.eval()

        #print('loading SDF model...')
        model.load_state_dict(load_checkpoint(name, device))
        mesh = marching_cubes_mesh(
            pc=pc,
            model=model,
            batch_size=4096,
            grid_size=grid_res, # increase to 128 for resolution used in evals
            progress=True,
        )
        
        # TODO: return mesh
    
    elif method == 'dmtet':
        if isinstance(pc, str):
            pc = PointCloud.load(pc)
        points = pc.coords
        center = (points.max(0)[0] + points.min(0)[0]) / 2
        max_l = (points.max(0)[0] - points.min(0)[0]).max()
        points = ((points - center) / max_l)* 0.9
        points = torch.tensor(points).to(device)
        
        if points.shape[0] > 100000:
            idx = list(range(points.shape[0]))
            np.random.shuffle(idx)
            idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)    
            points = points[idx]

        # The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
        center = (points.max(0)[0] + points.min(0)[0]) / 2
        max_l = (points.max(0)[0] - points.min(0)[0]).max()
        points = ((points - center) / max_l)* 0.9
        # timelapse.add_pointcloud_batch(category='input',
        #                              pointcloud_list=[points.cpu()], points_type = "usd_geom_points")
        tet_verts = torch.tensor(np.load('util/dmtet/samples/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)
        tets = torch.tensor(([np.load('util/dmtet/samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)
        
        # Initialize model and create optimizer
        model = Decoder(multires=multires).to(device)
        model.pre_train_sphere(1000)
        
        vars = [p for _, p in model.named_parameters()]
        optimizer = torch.optim.Adam(vars, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time
        for it in range(iterations):
            pred = model(tet_verts) # predict SDF and per-vertex deformation
            sdf, deform = pred[:, 0], pred[:, 1:]
            verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
            mesh_verts, mesh_faces = marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
            mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

            loss = loss_f(mesh_verts, mesh_faces, points, it, iterations, laplacian_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (it) % save_every == 0 or it == (iterations - 1): 
                print('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
                # save reconstructed mesh
                # timelapse.add_mesh_batch(
                #     iteration=it+1,
                #     category='extracted_mesh',
                #     vertices_list=[mesh_verts.cpu()],
                #     faces_list=[mesh_faces.cpu()]
                # )
                #if it == 5000:
                #    continue
                # return
            return mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy()
        
        
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it, laplacian_weight, iterations):
    pred_points = sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer