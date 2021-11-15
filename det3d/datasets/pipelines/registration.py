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
class Registration(object):
    def __init__(self, radius, debug=False):
        self.debug = debug
        self.radius = radius

    def __call__(self, res, info):
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            points = res['points'][gp_edges[1]]
            seq = res['lidar_sequence']
            colors = torch.randn(num_graphs, 3)
            frame_colors = torch.randn(len(seq.frames), 3)
            ps_p = vis.pointcloud('points', points[:, :3])
            ps_p.add_color_quantity('frame', frame_colors[points[:, -1].long()], enabled=True)
            ps_p.add_color_quantity('graph', colors[gp_edges[0]], enabled=True)
            graph_centers = scatter(points[:, :3], gp_edges[0], dim=0,
                                    dim_size=num_graphs, reduce='mean')
            vis.pointcloud('graph centers', graph_centers, radius=10e-4)
            print(f'find connected components: time={end_time-start_time:.4f}')
            import ipdb; ipdb.set_trace()
            vis.show()
