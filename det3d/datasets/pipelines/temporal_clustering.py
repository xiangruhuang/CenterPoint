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
                 radius=1.0):
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.radius = radius

    def find_clusters(self, voxels, voxels_velo):
        e0, e1 = voxel_graph(voxels, voxels, self.voxel_size, 0, 64).T.long()
        import ipdb; ipdb.set_trace()
        dv = (voxels_velo[e0] - voxels_velo[e1]).norm(p=2, dim=-1)
        dp = (voxels[e0] - voxels[e1]).norm(p=2, dim=-1)
        mask = (dv < self.dv_threshold) & (dp < self.dp_threshold)
        e0, e1 = e0[mask], e1[mask]
        num_voxels = voxels.shape[0]
        # find graphs
        A = csr_matrix((torch.ones_like(e0), (e0, e1)),
                       shape=(num_voxels, num_voxels))
        num_graphs, graph_idx = connected_components(A, directed=False)

        return num_graphs, graph_idx

    def __call__(self, res, info):
        voxels = res['voxels']
        voxels_velo = res['voxels_velo']
        
        num_graphs, graph_idx = self.find_clusters(voxels, voxels_velo)
        res['num_graphs'] = num_graphs
        res['graph_idx'] = graph_idx

        return res, info
