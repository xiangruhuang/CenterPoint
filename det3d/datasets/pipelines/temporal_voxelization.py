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
    def __init__(self, voxel_size, velocity=False,
                 filter_by_min_num_points=0, debug=False):
        self.voxel_size = voxel_size
        self.debug=debug
        self.velocity = velocity
        self.filter_by_min_num_points = filter_by_min_num_points
    
    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        frames = seq.frames
        
        voxel_size = torch.tensor(self.voxel_size, dtype=torch.float32)
        points, normals = seq.points4d(), seq.normals()
        points = torch.tensor(points, dtype=torch.float32)
        
        vp_edges = voxelization(points, voxel_size, False)[0].T.long()
        num_voxels = vp_edges[0].max() + 1
        res['vp_edges'] = vp_edges
        res['voxels'] = scatter(points[vp_edges[1]], vp_edges[0],
                                reduce='mean', dim=0, dim_size=num_voxels)
        if self.filter_by_min_num_points > 0:
            ev, ep = vp_edges
            deg = scatter(torch.ones_like(ep), ev,
                          reduce='sum', dim=0, dim_size=num_voxels)
            is_valid_voxel = (deg >= self.filter_by_min_num_points)
            new_num_voxels = is_valid_voxel.long().sum().item()
            new_voxel_index = torch.zeros(num_voxels, dtype=torch.long)-1
            new_voxel_index[is_valid_voxel] = torch.arange(new_num_voxels,
                                                           dtype=torch.long)
            is_valid_edge = is_valid_voxel[ev]
            is_valid_point = torch.zeros(points.shape[0], dtype=torch.bool)
            is_valid_point[ep] = is_valid_edge
            new_num_points = is_valid_point.long().sum().item()
            new_point_index = torch.zeros(points.shape[0], dtype=torch.long)-1
            new_point_index[is_valid_point] = torch.arange(new_num_points,
                                                           dtype=torch.long)
            
            seq.filter_points(is_valid_point)

            ep = new_point_index[ep[is_valid_edge]]
            ev = new_voxel_index[ev[is_valid_edge]]
            vp_edges = torch.stack([ev, ep], dim=0)
            res['vp_edges'] = vp_edges
            res['voxels'] = res['voxels'][is_valid_voxel]
            num_voxels = new_num_voxels
            if self.debug:
                from det3d.core.utils.visualization import Visualizer
                vis = Visualizer([], [])
                vis.pointcloud('points', seq.points4d()[:, :3])
                vis.pointcloud('voxels', res['voxels'][:, :3])
                vis.show()


        if self.velocity:
            velocity = seq.velocity()
            velocity = torch.tensor(velocity, dtype=torch.float32)
            ev, ep = vp_edges
            voxels_velo = scatter(velocity[ep], ev, dim=0,
                                  dim_size=num_voxels, reduce='mean')
            res['voxels_velo'] = voxels_velo

        end_time = time.time()
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            #ps_p = vis.pointcloud('points', res['points'][:, :3])
            #vis.pc_scalar('points', 'frame % 2', res['points'][:, -1].numpy() % 2)
            vis.boxes('box-original', seq.corners(), seq.classes())
            ps_v = vis.pointcloud('voxels', res['voxels'][:, :3])
            ps_v.add_scalar_quantity('frame % 2', res['voxels'][:, -1].numpy() % 2)
            #vis.pc_scalar('voxels', 'frame % 2', res['voxels'][:, -1].numpy() % 2)
            #torch.save(res['voxels'], 'voxels.pt')

            print(f'temporal voxelization: time={end_time-start_time:.4f}')
            #vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/temporal_voxelization.pth')
            vis.show()

        return res, info
