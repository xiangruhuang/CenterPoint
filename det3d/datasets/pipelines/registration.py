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

from det3d.solver.arap import MultiFrameARAP as Solver

@PIPELINES.register_module
class Registration(object):
    def __init__(self, radius, debug=False):
        self.debug = debug
        self.radius = radius

    def register(self, points, normals, voxels, vp_edges,
                 voxel_size=[0.6, 0.6, 0.6, 1]):
        """
        Args:
            points (torch.Tensor, [N, 4])
            normals (torch.Tensor, [N, 3])
            voxels (torch.Tensor, [V, 4])
            vp_edges 
            voxel_size

        Returns:
            
        """
        solver = Solver(points, normals, voxels, vp_edges, voxel_size)
        solver.solve()

    def __call__(self, res, info):
        # initialization
        seq = res['lidar_sequence']
        num_frames = len(seq.frames)
        frame_window_size = 5
        points4d = res['points']
        normals4d = res['normals']
        voxels4d = res['voxels']
        vp_edges_all = res['vp_edges']
        point_offset, voxel_offset = 0, 0
        point_index_pool = torch.zeros(points4d.shape[0], dtype=torch.long)
        voxel_index_pool = torch.zeros(voxels4d.shape[0], dtype=torch.long)
        
        for i in range(num_frames-frame_window_size):
            """ pre-processing:
                for a D-frame window:
                1. find point and voxels with frame id between [i, i+D)
                2. update edge indices between selected points & voxels
            """
            end_frame = i + frame_window_size
            point_mask = (points4d[:, -1] >= i) & (points4d[:, -1] < end_frame)
            point_indices = torch.where(point_mask)[0]
            point_index_pool[point_indices] = torch.arange(point_indices.shape[0])
            points = points4d[point_indices]
            normals = normals4d[point_mask]
            voxel_mask = (voxels4d[:, -1] >= i) & (voxels4d[:, -1] < end_frame)
            voxel_indices = torch.where(voxel_mask)[0]
            voxel_index_pool[voxel_indices] = torch.arange(voxel_indices.shape[0])
            voxels = voxels4d[voxel_indices]
            edge_mask = (voxels4d[vp_edges_all[0], -1] >= i) & \
                        (voxels4d[vp_edges_all[0], -1] < end_frame)
            vp_edges = vp_edges_all[:, edge_mask]
            vp_edges[0] = voxel_index_pool[vp_edges[0]]
            vp_edges[1] = point_index_pool[vp_edges[1]]

            # Debug and visualization
            if self.debug:
                from det3d.core.utils.visualization import Visualizer
                vis = Visualizer([], [])
                frame_colors = torch.randn(len(seq.frames), 3)
                ps_p = vis.pointcloud('points-before', points[:, :3])
                corners = seq.corners(i, i+frame_window_size)
                classes = seq.classes(i, i+frame_window_size)
                vis.boxes('box', corners, classes)
                vis.pc_color('points-before', 'frame',
                    frame_colors[points[:, -1].long()],
                    )
                vis.pc_scalar(
                    'points-before',
                    'frame % 2',
                    points[:, -1].long() % 2,
                    )
                vis.pc_scalar(
                    'points-before',
                    'frame',
                    points[:, -1].long(),
                    )
                #vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/registration.pth')
                vis.show()
            
            # register from frame i to frame i+d
            self.register(points, normals, voxels, vp_edges)
            
            # post processing
            voxel_offset += (voxels4d[:, -1] == i).long().sum()
            point_offset += (points4d[:, -1] == i).long().sum()
                
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            import ipdb; ipdb.set_trace()
            vis = Visualizer([], [])
            points = res['points'][gp_edges[1]]
            seq = res['lidar_sequence']
            colors = torch.randn(num_graphs, 3)
            frame_colors = torch.randn(len(seq.frames), 3)
            ps_p = vis.pointcloud('points', points[:, :3])
            ps_p.add_color_quantity('frame', frame_colors[points[:, -1].long()],
                enabled=True)
            ps_p.add_color_quantity('graph', colors[gp_edges[0]], enabled=True)
            graph_centers = scatter(points[:, :3], gp_edges[0], dim=0,
                                    dim_size=num_graphs, reduce='mean')
            vis.pointcloud('graph centers', graph_centers, radius=10e-4)
            print(f'find connected components: time={end_time-start_time:.4f}')
            #vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/registration2.pth')
            vis.show()
            #vis.show()

