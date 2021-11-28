import open3d as o3d
import numpy as np
import torch
from torch_scatter import scatter
from ..registry import PIPELINES
from det3d.ops.primitives.primitives_cpu import (
    voxelization,
    voxel_graph,
    query_point_correspondence as query_corres,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

@PIPELINES.register_module
class FindConnectedComponents(object):
    def __init__(self, radius, debug=False, granularity='voxels',
                 max_num_neighbors=32):
        self.radius = radius
        voxel_size = [radius, radius, radius, 1.0]
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.debug=debug
        self.granularity = granularity
        assert granularity in ['points', 'voxels']
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, res, info):
        import time
        start_time = time.time()
        if self.granularity == 'voxels':
            if res.get('voxels', None) is None:
                raise ValueError('voxels not present in res,'\
                                 'use TemporalVoxelization First')
            else:
                nodes = res['voxels']
        else:
            if res.get('points', None) is not None:
                nodes = res['points']
            else:
                seq = res['lidar_sequence']
                nodes = seq.points4d()
                nodes = torch.tensor(nodes, dtype=torch.float32)
        e0, e1 = voxel_graph(nodes, nodes, self.voxel_size.float().cpu(),
                             0, self.max_num_neighbors).T.long()
        dist = (nodes[e0, :3] - nodes[e1, :3]).norm(p=2, dim=-1)
        valid_mask = (e0 != e1) & (dist < self.radius)
        e0 = e0[valid_mask]
        e1 = e1[valid_mask]
        num_nodes = nodes.shape[0]
        A = csr_matrix((torch.ones_like(e0), (e0, e1)),
                       shape=(num_nodes, num_nodes))
        num_graphs, graph_idx = connected_components(A, directed=False)
        graph_idx = torch.tensor(graph_idx, dtype=torch.long)
        vp_edges = res['vp_edges']
        if self.granularity == 'voxels':
            graph_idx = graph_idx[vp_edges[0]]
            gp_edges = torch.stack([graph_idx, vp_edges[1]],
                                   dim=0).long()
        else:
            gp_edges = torch.stack([graph_idx,
                                    torch.arange(graph_idx.shape[0])],
                                   dim=0).long()

        res['gp_edges'] = gp_edges
        end_time = time.time()

        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            seq = res['lidar_sequence']
            points = torch.tensor(seq.points4d(), dtype=torch.float32)
            import ipdb; ipdb.set_trace()
            points = points[gp_edges[1]]
            colors = torch.randn(num_graphs, 3)
            frame_colors = torch.randn(len(seq.frames), 3)
            ps_p = vis.pointcloud('points', points[:, :3])
            ps_p.add_color_quantity('frame', frame_colors[points[:, -1].long()],
                                    enabled=True)
            ps_p.add_color_quantity('graph', colors[gp_edges[0]], enabled=True)
            mask0 = (points[:, -1] == 0)
            mask1 = (points[:, -1] == 1)
            p0, p1 = points[mask0], points[mask1]
            ps_p0 = vis.pointcloud('p0', p0[:, :3])
            ps_p1 = vis.pointcloud('p1', p1[:, :3])
            ps_p0.add_color_quantity('graph', colors[gp_edges[0]][mask0],
                                     enabled=True)
            ps_p1.add_color_quantity('graph', colors[gp_edges[0]][mask1],
                                     enabled=True)
            graph_centers = scatter(points[:, :3], gp_edges[0], dim=0,
                                    dim_size=num_graphs, reduce='mean')
            vis.pointcloud('graph centers', graph_centers, radius=10e-4)
            print(f'find connected components: time={end_time-start_time:.4f}')
            vis.show()

        return res, info
        
