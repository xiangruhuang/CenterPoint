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
   
    def curvenetwork(self, name, nodes, edges):
        return ps.register_curve_network(name, nodes, edges, radius=self.radius)

    def pointcloud(self, name, pointcloud, color=None, radius=None):
        """Visualize non-zero entries of heat map on 3D point cloud.
            point cloud (torch.Tensor, [N, 3])
        """
        if radius is None:
            radius = self.radius
        if color is None:
            return ps.register_point_cloud(name, pointcloud, radius=self.radius)
        else:
            return ps.register_point_cloud(
                name, pointcloud, radius=self.radius, color=color
                )
    
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

    def boxes(self, name, corners, labels=None):
        """
            corners (shape=[N, 8, 3]):
            labels (shape=[N])
        """
        edges = [[0, 1], [0, 3], [0, 4], [1, 2],
                 [1, 5], [2, 3], [2, 6], [3, 7],
                 [4, 5], [4, 7], [5, 6], [6, 7]]
        N = corners.shape[0]
        edges = np.array(edges) # [12, 2]
        edges = np.repeat(edges[np.newaxis, ...], N, axis=0) # [N, 12, 2]
        offset = np.arange(N)[..., np.newaxis, np.newaxis]*8 # [N, 1, 1]
        edges = edges + offset
        ps_box = ps.register_curve_network(
                     name, corners.reshape(-1, 3),
                     edges.reshape(-1, 2), radius=2e-4
                 )
        if labels is not None:
            # R->Car, G->Ped, B->Cyc
            colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,0]])
            labels = np.repeat(labels[:, np.newaxis], 8, axis=-1).reshape(-1)
            ps_box.add_color_quantity('class', colors[labels],
                                      defined_on='nodes', enabled=True)
        return ps_box

    def heatmap(self, name, heatmap, color=True, threshold=0.1, radius=2e-4, **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
            heatmap (torch.Tensor, [W, H])
        """
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        indices = list(torch.where(heatmap > threshold))
        heights = heatmap[indices]
        indices = indices[::-1]
        for i in range(2):
            indices[i] = indices[i] * self.size_factor * self.voxel_size[i] + self.pc_range[i]

        coors = torch.stack([*indices, heights], dim=-1)
        ps_p = ps.register_point_cloud(name, coors, radius=radius, **kwargs)
        if color:
            ps_p.add_scalar_quantity("height", (coors[:, -1]), enabled=True) 

        return ps_p

    def show(self):
        ps.show()

class SeqVisualizer(Visualizer):
    def __init__(self,
                 voxel_size,
                 pc_range,
                 size_factor=8,
                 radius=2e-4,
                 **kwargs):
        super(SeqVisualizer, self).__init__(
            voxel_size,
            pc_range,
            size_factor,
            radius,
            **kwargs)
        self.frames = {}

    def add_tpoints(self, prefix, tpoints):
        min_f = tpoints[:, -1].min().long()
        max_f = tpoints[:, -1].max().long()+1
        for i in range(min_f, max_f):
            frame_points = tpoints[tpoints[:, -1] == i, :3]
            self.pointcloud(f'{prefix}-{i}', frame_points)
        
    def add_frame(self, timestamp, **frame):
        self.frames[timestamp] = frame

    def visualize_frames(self, start, end):
        self.clear()
        origins = []
        for ts, frame in self.frames.items():
            if ts > end or ts < start:
                continue
            frame = self.frames[ts]
            if frame.get('points', None) is not None:
                ps_p = self.pointcloud(
                    f'points-{ts}', frame['points'], frame['color'])
                if frame.get('num_clusters', None) is not None:
                    num_clusters = frame['num_clusters']
                    cluster_ids = frame['cluster_ids']
                    colors = np.random.randn(num_clusters, 3)
                    ps_p.add_color_quantity(
                        'cluster', colors[cluster_ids], enabled=True)
            if frame.get('boxes', None) is not None:
                self.boxes(f'boxes-{ts}', frame['boxes'], frame['labels'])
            if frame.get('origin', None) is not None:
                origins.append((ts, frame['origin']))


        origins = sorted(origins, key=lambda x: x[0])
        origins = np.array([ori[1] for ori in origins])
        num_frames = origins.shape[0]
        origin_edges = np.stack([np.arange(0, num_frames-1),
                                 np.arange(1, num_frames)], axis=-1)
        self.curvenetwork('origins', origins, origin_edges)
        self.show() 
