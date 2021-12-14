import time
import os
from ..registry import PIPELINES
import torch
import numpy as np
from det3d.ops.primitives.primitives_cpu import (
    voxel_graph, voxelization,
    query_point_correspondence as query_corres
)
from torch_scatter import scatter
from det3d.core.utils.visualization import Visualizer
from .kalman_filter import StatelessKalmanFilter

class KalmanGroupTracker(object):
    def __init__(self,
                 ht,
                 kf_config,
                 threshold=1.0,
                 acc_threshold=1.0,
                 voxel_size=[0.6, 0.6, 0.6, 1],
                 corres_voxel_size=[2.5, 2.5, 2.5, 1],
                 debug=False,
                 ):
        self.kf_config = kf_config
        self.ht = ht
        self.threshold = threshold
        self.acc_threshold = acc_threshold
        self.debug = debug
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.corres_voxel_size = torch.tensor(corres_voxel_size,
                                              dtype=torch.float32).cuda()
        self.kt = StatelessKalmanFilter(**kf_config)

    def set_points(self, points):
        self.points = points.clone().cuda()
        self.pc_range = torch.cat([points.min(0)[0]-3,
                                   points.max(0)[0]+3], dim=0).cuda()
        self.num_frames = points[:, -1].max().long()+1
        self.frame_indices = []
        self.frames = []
        for i in range(self.num_frames):
            frame_index = torch.where(self.points[:, -1]==i)[0]
            self.frame_indices.append(frame_index)
            self.frames.append(self.points[frame_index])
        # prepare hash map (save time)
        self.ht.hash_into_gpu(self.points,
                              self.corres_voxel_size,
                              self.pc_range)

    def register(self, p, q, graph_idx, num_graphs,
                 corres_size, valid_mask):
        assert p.dtype == torch.float64
        assert q.dtype == torch.float64
        p_centers = scatter(p, graph_idx, dim=0, dim_size=num_graphs,
                            reduce='mean')
        q_centers = scatter(q, graph_idx, dim=0, dim_size=num_graphs,
                            reduce='mean')
        p = p - p_centers[graph_idx]
        q = q - q_centers[graph_idx]
        shift = q_centers - p_centers
        R = torch.eye(3, dtype=torch.float64).repeat(num_graphs, 1, 1).cuda()

        for itr in range(3):
            diff = (q - (R[graph_idx] @ p.unsqueeze(-1)).squeeze(-1) )
            t = scatter(diff, graph_idx, dim=0, dim_size=num_graphs,
                        reduce='sum')
            t[valid_mask] /= corres_size[valid_mask].unsqueeze(-1)
            M = p[:, :2].unsqueeze(-1) @ (q-t[graph_idx])[:, :2].unsqueeze(-2)
            M = scatter(M.view(-1, 4), graph_idx, dim=0,
                        dim_size=num_graphs, reduce='sum').view(-1, 2, 2)
            M[valid_mask] /= corres_size[valid_mask].unsqueeze(-1).unsqueeze(-1)
            U, S, V = M[valid_mask].svd()
            mask = (V @ U.transpose(1, 2)).det() < 0
            V[mask, -1] *= -1
            R[valid_mask, :2, :2] = V @ U.transpose(1, 2)
            diff = q-t[graph_idx] - (R[graph_idx] @ p.unsqueeze(-1)).squeeze(-1)
            error = scatter(diff.norm(p=2, dim=-1), graph_idx, reduce='sum',
                            dim=0, dim_size=num_graphs)
            error[valid_mask] /= corres_size[valid_mask]
        error[valid_mask == False] = 1e10
        t = t + shift

        return R, t, error

    def track_graphs(self,
                     moving_points,
                     eg,
                     graph_size,
                     temporal_dir):
        num_graphs = eg.max().item()+1
        
        # find correspondence
        e_moving, e_static = self.ht.find_corres_step2(
                                 moving_points.float(), temporal_dir)

        # filter bad correspondence graphs
        corres_size = scatter(torch.ones_like(e_moving),
                              eg[e_moving], dim=0,
                              dim_size=num_graphs, reduce='sum').double()
        valid_mask = (corres_size > 2) & (corres_size*2 >= graph_size)
        
        # exclude zero-correspondence graphs via `valid_mask`.
        gwise_R, gwise_t, gwise_error = \
            self.register(moving_points[e_moving, :3],
                          self.points[e_static, :3].double(),
                          eg[e_moving], num_graphs,
                          corres_size, valid_mask)
        
        return gwise_R, gwise_t, gwise_error
