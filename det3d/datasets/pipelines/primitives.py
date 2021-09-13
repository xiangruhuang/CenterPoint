import torch
import numpy as np
from torch_scatter import scatter
from sklearn.neighbors import NearestNeighbors as NN

from det3d.core.input.dynamic_voxel import DynamicVoxel
from ..registry import PIPELINES

@PIPELINES.register_module
class ExtractPrimitives(object):
    def __init__(self,
                 voxel_size,
                 elim_th=0.1,
                 visualize=True,
                 ):
        self.voxel_size = voxel_size
        self.visualize = visualize
        self.elim_th = elim_th
        self.pts_voxel_layer = DynamicVoxel(
            voxel_size=self.voxel_size,
        )
    
    def find_unique_voxels(self, coors):
        """Find unique voxels.

        Args:
            coors (torch.Tensor, shape=[N, 3]): voxel coordinates per point

        Returns:
            num_voxels (int): number of unique voxels
            voxel_indices (shape=[N]): unique voxel ids for each point
            num_points (shape=[M]): number of points in each voxel

        """
        # flip back from [z, y, x] to [x, y, z]
        voxel_dims = torch.flip(coors.max(0)[0] + 1, dims=[0])
        ids = coors[:, 2] * voxel_dims[1]
        ids = (ids + coors[:, 1]) * voxel_dims[2] + coors[:, 0]
        
        unique_voxel_ids, voxel_indices, num_points = \
            torch.unique(ids, return_inverse=True, return_counts=True)
        num_voxels = unique_voxel_ids.shape[0]
        return num_voxels, voxel_indices, num_points

    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

        v1 = eigvecs[:, :, 1]
        v2 = eigvecs[:, :, 2]
        e1 = eigvals[:, 1].unsqueeze(-1).sqrt()
        e2 = eigvals[:, 2].unsqueeze(-1).sqrt()
        corners = []
        for d1 in [-1, 1]:
            for d2 in [-1, 1]:
                corners.append(centers + d1*v1*e1 + d2*v2*e2)
        num_voxels = centers.shape[0]
        corners = torch.stack(corners, dim=1) # [M, 4, 3]
        faces = [[0, 1, 3, 2]]
        faces = torch.as_tensor(faces, dtype=torch.long)
        faces = faces.repeat(num_voxels, 1, 1)
        faces += torch.arange(num_voxels).unsqueeze(-1).unsqueeze(-1)*4

        return corners.view(-1, 3), faces.view(-1, 4)

    def plane_fitting(self, points, voxel_indices, num_points,
                      sigma=0.07, eps=1e-6):
        """Fit planes iteratively following RANSAC.

        Args:
            points (shape=[N, 3]): points
            voxel_indices (shape=[N]): voxel indices per point.
            num_points (shape=[M]): number of points in each voxel.

        Returns:
            confidence (shape=[M]): per voxel confidence
            planes (shape=[M, 3]): per voxel normals
        """
        
        num_voxels = voxel_indices.max()+1
        vs = np.array(self.pts_voxel_layer.voxel_size)
        sigma2 = (vs * vs).sum()*sigma*sigma
        weights = torch.ones_like(voxel_indices).float()
        for itr in range(10):
            # compute weighted centers
            weights_e = weights.unsqueeze(-1)
            centers = scatter(points*weights_e, voxel_indices, dim=0,
                              reduce='add', dim_size=num_voxels)
            w_sum = scatter(weights_e, voxel_indices, dim=0,
                            reduce='add', dim_size=num_voxels)
            centers = centers / w_sum
            xyz_centered = points - centers[voxel_indices]
            
            # compute normals
            weights_e = weights_e.unsqueeze(-1)
            ppT = (xyz_centered.unsqueeze(-1) @ xyz_centered.unsqueeze(-2))
            V = scatter(ppT*weights_e, voxel_indices, dim=0,
                        reduce='add', dim_size=num_voxels)
            V = V / w_sum.unsqueeze(-1)
            eigvals, eigvecs = np.linalg.eigh(V)
            normals = eigvecs[:, :, 0]

            # update weights
            residual = np.abs((xyz_centered * normals[voxel_indices]).sum(-1))
            weights = sigma2/(sigma2+residual*residual+eps)
        
        #d1 = (xyz_centered * eigvecs[:, :, 1][voxel_indices]).sum(-1)
        #d2 = (xyz_centered * eigvecs[:, :, 2][voxel_indices]).sum(-1)

        #d1 = d1[eliminate == False]
        #d2 = d2[eliminate == False]
        #voxel_indices_s = voxel_indices[eliminate == False]
        #dims = []
        #dims.append(
        #    scatter(d1, voxel_indices_s, dim=0,
        #        reduce='min', dim_size=num_voxels))
        #dims.append(
        #    scatter(d1, voxel_indices_s, dim=0,
        #        reduce='max', dim_size=num_voxels))
        #dims.append(
        #    scatter(d2, voxel_indices_s, dim=0,
        #        reduce='min', dim_size=num_voxels))
        #dims.append(
        #    scatter(d2, voxel_indices_s, dim=0,
        #        reduce='max', dim_size=num_voxels))

        eliminate = (residual < self.elim_th)
        normals = torch.as_tensor(normals)
        eigvecs = torch.as_tensor(eigvecs)
        eigvals = torch.as_tensor(eigvals)
        conf_mean = scatter(
            eliminate.float(), voxel_indices, dim=0,
            reduce='mean', dim_size=num_voxels)
        conf_sum = scatter(
            eliminate.float(), voxel_indices, dim=0,
            reduce='sum', dim_size=num_voxels)
        thresholds = [(0.7, 10), (0.8, 7), (0.9, 4)]
        valid_plane_mask = conf_mean < 0
        for mean_th, sum_th in thresholds:
            valid_plane_mask |= ((conf_mean >= mean_th) & (conf_sum >= sum_th))
        surfels = torch.cat([centers, normals, eigvals[:, 1:],
                             eigvecs[:, :, 1], eigvecs[:, :, 2]], dim=-1)
        valid_surfels = surfels[valid_plane_mask]
        valid_centers = centers[valid_plane_mask]
        valid_normals = normals[valid_plane_mask]
        tree = NN(n_neighbors=1).fit(valid_centers)
        dists, indices = tree.kneighbors(points)
        dists, indices = dists[:, 0], indices[:, 0]
        
        eliminate = eliminate & valid_plane_mask[voxel_indices]
        remain_points = points[eliminate == False]
        residual = residual[eliminate == False]

        if self.visualize:
            corners, faces = self.get_meshes(
                centers[valid_plane_mask],
                eigvals[valid_plane_mask],
                eigvecs[valid_plane_mask])
            import polyscope as ps
            ps.set_up_dir('z_up'); ps.init()
            ps_p = ps.register_point_cloud('points', remain_points, radius=2e-4)
            ps_p.add_scalar_quantity('residual', residual)
            ps.register_surface_mesh('planes', corners.numpy(), faces.numpy())

            ps.show()
            
        return valid_surfels, (eliminate == False)

    def __call__(self, results, info):
        points = results['lidar']['points']
        coors = self.pts_voxel_layer(points).astype(np.int32)
        coors = torch.as_tensor(coors)
        num_voxels, voxel_indices, num_points = self.find_unique_voxels(coors)
        # find voxel-wise geometric centers
        points_xyz = torch.from_numpy(points[:, :3])
        centers = scatter(points_xyz, voxel_indices, dim=0,
                          reduce='mean', dim_size=num_voxels)
        xyz_centered = points_xyz - centers[voxel_indices]
        # aggregate (p p^T) for each point vector p \in R^3
        ppT = (xyz_centered.unsqueeze(-1) @ xyz_centered.unsqueeze(-2))
        V = scatter(ppT, voxel_indices, dim=0,
                    reduce='mean', dim_size=num_voxels)
        eigvals = np.linalg.eigvalsh(V)
        
        # find possible planes
        surfels, mask = self.plane_fitting(
            points_xyz, voxel_indices, num_points)
       
        results['lidar']['planes'] = surfels
        results['lidar']['points'] = points[mask]
         
        return results, info
