import open3d as o3d
import numpy as np
import torch
from torch_scatter import scatter
from ..registry import PIPELINES
from det3d.ops.primitives.primitives_cpu import (
    voxelization, 
)

@PIPELINES.register_module
class TemporalVoxelization(object):
    """Group points in all frame in spatio-temporal (4D) voxels

    Args:
        res['frames']: frames
        voxel_size (torch.tensor, shape=[4]): spatio-temporal voxel size

    Returns:
        points (N, 4): (x,y,z,t) 4D points
        normals (N, 3): (nx, ny, nz) normals per point.
        voxels (V, 4): (x,y,z,t) 4D voxels
        vp_edges (2, E): (voxel, point) edges
    """
    def __init__(self, voxel_size, debug=False):
        self.voxel_size = voxel_size
        self.debug=debug
    
    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        frames = seq.frames
        voxel_size = torch.tensor(self.voxel_size, dtype=torch.float32)
        points, normals = seq.points4d(), seq.normals()
        points = torch.tensor(points, dtype=torch.float32)
        normals = torch.tensor(normals, dtype=torch.float32)
        res['points'] = points
        res['normals'] = normals 
        
        vp_edges = voxelization(points, voxel_size, False)[0].T.long()
        num_voxels = vp_edges[0].max() + 1
        res['vp_edges'] = vp_edges
        res['voxels'] = scatter(points[vp_edges[1]], vp_edges[0],
                                reduce='mean', dim=0, dim_size=num_voxels)
        end_time = time.time()
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            ps_p = vis.pointcloud('points', res['points'][:, :3])
            vis.pc_scalar('points', 'frame % 2', res['points'][:, -1].numpy() % 2)
            vis.boxes('box-original', seq.corners(), seq.classes())
            ps_v = vis.pointcloud('voxels', res['voxels'][:, :3])
            vis.pc_scalar('voxels', 'frame % 2', res['voxels'][:, -1].numpy() % 2)
            torch.save(res['voxels'], 'voxels.pt')

            print(f'temporal voxelization: time={end_time-start_time:.4f}')
            #vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/temporal_voxelization.pth')
            vis.show()

        return res, info
