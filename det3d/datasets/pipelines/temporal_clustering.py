from ..registry import PIPELINES
from det3d.ops.primitives.primitives_cpu import (
    voxel_graph,
)
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

@PIPELINES.register_module
class TemporalClustering(object):
    """ Basically a Kalman Filter"""
    def __init__(self, voxel_size,
                 radius=1.0, debug=False):
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.radius = 1.0
        self.debug = debug

    def find_clusters(self, voxels, voxels_velo):
        e0, e1 = voxel_graph(voxels, voxels, self.voxel_size, 0, 64).T.long()
        dp = (voxels[e0] - voxels[e1]).norm(p=2, dim=-1)
        mask = dp < self.radius
        e0, e1 = e0[mask], e1[mask]
        num_voxels = voxels.shape[0]
        # find graphs
        A = csr_matrix((torch.ones_like(e0), (e0, e1)),
                       shape=(num_voxels, num_voxels))
        num_graphs, graph_idx = connected_components(A, directed=False)

        return num_graphs, graph_idx

    def __call__(self, res, info):
        voxels = res['voxels']
        num_graphs, graph_idx = self.find_clusters(voxels)
        res['num_graphs'] = num_graphs
        res['graph_idx'] = graph_idx
        if self.debug:
            import numpy as np
            from torch_scatter import scatter
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            seq = res['lidar_sequence']
            points = seq.points4d()
            mask_v0 = voxels[:, -1] == 0
            mask_p0 = points[:, -1] == 0
            colors = torch.randn(num_graphs, 3)
            v0, p0 = voxels[mask_v0], points[mask_p0]
            ps_v = vis.pointcloud('voxels', voxels[:, :3], enabled=False)
            ps_v0 = vis.pointcloud('v0', v0[:, :3])
            ps_p0 = vis.pointcloud('p0', p0[:, :3])
            ps_v0.add_color_quantity('graph', colors[graph_idx[mask_v0]])
            ev, ep = res['vp_edges']
            graph_idx_by_point = scatter(torch.tensor(graph_idx)[ev], ep,
                                         reduce='max', dim=0,
                                         dim_size=points.shape[0]).numpy()
            ps_p0.add_color_quantity('graph', colors[graph_idx_by_point[mask_p0]])

            vis.show()

        return res, info
