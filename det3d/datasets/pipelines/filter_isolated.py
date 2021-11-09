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
class FilterIsolatedPoints(object):
    def __init__(self, radius = 0.7, threshold=3, debug=False):
        voxel_size = [radius, radius, radius, 1.0]
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.debug = debug
        self.radius = radius
        self.threshold = threshold

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        points = seq.points4d()
        points = torch.tensor(points, dtype=torch.float32)
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            ps_p = vis.pointcloud('points-original', points[:, :3])
        e0, e1 = voxelization(points, self.voxel_size.float().cpu(),
                              False)[0].T.long()
        num_voxels = e0.max()+1
        deg = scatter(torch.ones_like(e0), e0, dim=0,
                      dim_size=num_voxels, reduce='sum')
        valid_mask = (deg >= self.threshold)[e0]
        for f in seq.frames:
            valid_mask_f = valid_mask[points[:, -1].long() == f.frame_id]
            f.filter(valid_mask_f)
        res['lidar_sequence'] = seq
        end_time = time.time()
       
        if self.debug:
            points = seq.points4d()
            ps_p = vis.pointcloud('points', points[:, :3])
            print(f'filter isolated points: time={end_time-start_time:.4f}')
            vis.show()

        return res, info
