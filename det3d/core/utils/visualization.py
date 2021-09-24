import polyscope as ps
import torch
import numpy as np

class Visualizer:
    def __init__(self,
                 voxel_size,
                 pc_range,
                 size_factor=8,
                 radius=2e-4):
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.size_factor = size_factor
        self.radius = radius
        ps.set_up_dir('z_up')
        ps.init()

    def clear(self):
        ps.remove_all_structures()
    
    def pointcloud(self, name, pointcloud):
        """Visualize non-zero entries of heat map on 3D point cloud.
            point cloud (torch.Tensor, [N, 3])
        """
        return ps.register_point_cloud(name, pointcloud, radius=self.radius)
    
    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

        v1 = eigvecs[:, :3]
        v2 = eigvecs[:, 3:]
        e1 = np.sqrt(eigvals[:, 0:1])
        e2 = np.sqrt(eigvals[:, 1:2])
        corners = []
        for d1 in [-1, 1]:
            for d2 in [-1, 1]:
                corners.append(centers + d1*v1*e1 + d2*v2*e2)
        num_voxels = centers.shape[0]
        corners = np.stack(corners, axis=1) # [M, 4, 3]
        faces = [0, 1, 3, 2]
        faces = np.array(faces, dtype=np.int32)
        faces = np.repeat(faces[np.newaxis, np.newaxis, ...], num_voxels, axis=0)
        faces += np.arange(num_voxels)[..., np.newaxis, np.newaxis]*4
        return corners.reshape(-1, 3), faces.reshape(-1, 4)
    
    def planes(self, name, planes):
        corners, faces = self.get_meshes(planes[:, :3], planes[:, 6:8], planes[:, 8:14])
        return ps.register_surface_mesh(name, corners, faces)

    def boxes(self, name, corners):
        """
            corners (shape=[N, 8, 3]):
        """
        edges = [[0, 1], [0, 3], [0, 4], [1, 2],
                 [1, 5], [2, 3], [2, 6], [3, 7],
                 [4, 5], [4, 7], [5, 6], [6, 7]]
        N = corners.shape[0]
        edges = np.array(edges) # [12, 2]
        edges = np.repeat(edges[np.newaxis, ...], N, axis=0) # [N, 12, 2]
        edges = edges + np.arange(N)[..., np.newaxis, np.newaxis]*8 # += [N, 1, 1]
        return ps.register_curve_network(
                   name, corners.reshape(-1, 3), edges.reshape(-1, 2), radius=4e-4
               )

    def heatmap(self, name, heatmap, color=True, threshold=0.1):
        """Visualize non-zero entries of heat map on 3D point cloud.
            heatmap (torch.Tensor, [W, H])
        """
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        indices = list(torch.where(heatmap > 0))
        heights = heatmap[indices]
        indices = indices[::-1]
        for i in range(2):
            indices[i] = indices[i] * self.size_factor * self.voxel_size[i] + self.pc_range[i]

        coors = torch.stack([*indices, heights], dim=-1)
        coors = coors[coors[:, 2] > threshold]
        ps_p = ps.register_point_cloud(name, coors, radius=self.radius*5)
        if color:
            ps_p.add_scalar_quantity("height", (coors[:, -1]+1e-6).log(), enabled=True) 

        return ps_p

    def show(self):
        ps.show()

