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
from PytorchHashmap.torch_hash import HashTable
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import glob
from .kalman_tracker import KalmanTracker
from .kalman_group_tracker import KalmanGroupTracker

@PIPELINES.register_module
class ObjTracking(object):
    """ Basically a Kalman Filter"""
    def __init__(self,
                 kf_config,
                 threshold=1.0,
                 acc_threshold=1.0,
                 reg_threshold=1.0,
                 angle_threshold=40,
                 min_travel_dist=3.0,
                 min_mean_velocity=0.3,
                 voxel_size=[0.6, 0.6, 0.6, 1],
                 corres_voxel_size=[2.5, 2.5, 2.5, 1],
                 min_velo=0.05,
                 velocity=True,
                 crop_points=True,
                 debug=False,
                 ):
        self.reg_threshold = reg_threshold
        self.angle_threshold = angle_threshold
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.min_travel_dist = min_travel_dist
        self.min_mean_velocity = min_mean_velocity
        self.min_velo = min_velo
        self.velocity = velocity
        self.crop_points = crop_points
        self.debug = debug
        self.ht = HashTable(40000000)
        self.tracker = KalmanGroupTracker(
                           self.ht, kf_config, threshold, acc_threshold,
                           voxel_size, corres_voxel_size, debug)
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            self.vis = Visualizer([], [])
        
    def check_trace_status(self, trace, centers):
        if len(trace) < 10:
            if self.debug:
                print(f'short trace, length={len(trace)}')
            return False
        lengths = (centers[1:, :3] - centers[:-1, :3]).norm(p=2, dim=-1)
        travel_dist = lengths.sum()
        mean_velo = (centers[0, :3] - centers[-1, :3]
                     ).norm(p=2, dim=-1) / lengths.shape[0]
        if travel_dist < self.min_travel_dist:
            if self.debug:
                print(f'not moving, traveled {travel_dist:.4f}')
            return False
        if mean_velo < self.min_mean_velocity:
            if self.debug:
                print(f'not moving, mean velo = {mean_velo:.4f}')
            return False
        return True

    def filter_graphs(self, points, graph_idx_by_point, num_point_range):
        num_points = graph_idx_by_point.shape[0]
        num_graphs = graph_idx_by_point.max().item()+1
        deg = scatter(torch.ones(num_points).long(), graph_idx_by_point,
                      dim=0, dim_size=num_graphs, reduce='sum')
        mask = (deg >= num_point_range[0]) & (deg < num_point_range[1])
        pointwise_mask = mask[graph_idx_by_point]
        
        new_graph_index = torch.zeros(num_graphs, dtype=torch.long) - 1
        new_graph_index[mask] = torch.arange(mask.sum()).long()
        graph_idx_by_point = new_graph_index[graph_idx_by_point]
        point_indices = torch.where(graph_idx_by_point != -1)[0]
        graph_idx_by_point = graph_idx_by_point[point_indices]
        
        return graph_idx_by_point, points[point_indices], num_graphs
    
    def translate(self, points, graph_idx, is_active_graph):
        """

        Returns:
            points
            graph_idx
        """
        is_active_point = is_active_graph[graph_idx]
        num_graphs = is_active_graph.shape[0]
        new_graph_index = torch.zeros(num_graphs, dtype=torch.long).cuda() - 1
        num_active_graphs = is_active_graph.long().sum().item()
        new_graph_index[is_active_graph] = torch.arange(num_active_graphs).long().cuda()

        return points[is_active_point], new_graph_index[graph_idx][is_active_point]

    def track_dir(self, points, graph_idx, temporal_dir,
                  transformations, trace_centers, visited):
        points = points.clone().double().cuda()
        graph_idx = graph_idx.clone().cuda()
        num_graphs = graph_idx.max().item() + 1
        graph_size = scatter(torch.ones_like(graph_idx), graph_idx,
                             dim=0, dim_size=num_graphs, reduce='sum')
        original_graph_indices = torch.arange(num_graphs,
                                              dtype=torch.long).cuda()
        centers = scatter(points[:, :3], graph_idx, dim=0,
                          dim_size=num_graphs, reduce='mean')
        last_centers = None
        num_existing_graphs = num_graphs
        gwise_frame_id = scatter(points[:, -1], graph_idx, dim=0,
                                 dim_size=num_existing_graphs,
                                 reduce='max').long()
        trace_centers[(torch.arange(num_graphs), gwise_frame_id)] = centers
        visited[(original_graph_indices, gwise_frame_id)] = True

        while True:
            start_time = time.time()
            gwise_R, gwise_t, gwise_error = \
                                self.tracker.track_graphs(
                                    points, graph_idx,
                                    graph_size[original_graph_indices],
                                    temporal_dir)

            # update transformations
            gwise_cos = gwise_R[:, 0, 0].clip(-1, 1)
            gwise_sin = gwise_R[:, 1, 0].clip(-1, 1)
            gwise_cossin = torch.stack([gwise_cos, gwise_sin], dim=-1)
            gwise_pose = torch.cat([gwise_t, gwise_cossin], dim=-1)

            is_active_graph = gwise_error < self.reg_threshold
            if is_active_graph.any() == False:
                break
            is_active_point = is_active_graph[graph_idx]
            pwise_t, pwise_R = gwise_t[graph_idx], gwise_R[graph_idx]
            if self.debug:
                ps_c = self.vis.pointcloud('graph_centers',
                                      centers[original_graph_indices, :3].cpu())
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=num_existing_graphs,
                                    reduce='mean')
            points[:, :3] -= gwise_centers[graph_idx]
            points[:, :3] = (pwise_R @ points[:, :3].unsqueeze(-1)).squeeze(-1) + pwise_t
            points[:, :3] += gwise_centers[graph_idx]
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=num_existing_graphs,
                                    reduce='mean')
            if self.debug:
                ps_c.add_vector_quantity('velocity',
                                         (gwise_centers \
                                          - centers[original_graph_indices]).cpu())

            # check acceleration
            if last_centers is not None:
                new_centers = scatter(points[:, :3], graph_idx, dim=0,
                                      dim_size=num_existing_graphs,
                                      reduce='mean')
                velocity = new_centers - centers[original_graph_indices]
                last_velocity = centers[original_graph_indices] \
                                - last_centers[original_graph_indices]
                acc = (last_velocity - velocity).norm(p=2, dim=-1)
                is_active_graph = (acc < self.tracker.acc_threshold) & is_active_graph
                if is_active_graph.any() == False:
                    break
            
            points, graph_idx = self.translate(points, graph_idx, is_active_graph)
            points[:, -1] += temporal_dir

            # update graph indices for next iteration
            original_graph_indices = original_graph_indices[is_active_graph]
            num_existing_graphs = original_graph_indices.shape[0]

            # update transformations, centers and visited
            gwise_frame_id = scatter(points[:, -1], graph_idx, dim=0,
                                     dim_size=num_existing_graphs,
                                     reduce='max').long()
            gwise_pose = gwise_pose[is_active_graph]

            visited[(original_graph_indices, gwise_frame_id)] = True
            transformations[(original_graph_indices, gwise_frame_id)] = gwise_pose
            trace_centers[(original_graph_indices, gwise_frame_id)] = \
                            scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=num_existing_graphs,
                                    reduce='mean')

            # update last centers
            last_centers = centers.clone()
            centers[original_graph_indices] = \
                    scatter(points[:, :3], graph_idx, dim=0,
                            dim_size=num_existing_graphs,
                            reduce='mean')
            print(f'num existing graphs={original_graph_indices.shape[0]}'\
                  f', time={time.time()-start_time}')

    def visualize_trace(self, graph_idx, points, visited, transformations,
                        num_selected_graphs, num_frames):
        if self.debug:
            print(f'num trace = {num_selected_graphs}')
        for gid in range(num_selected_graphs):
            mask_g = graph_idx == gid
            trace = [points[mask_g].cpu().clone()]
            for tdir in [-1, 1]:
                points_g = points[mask_g].cpu().clone().double()
                frame_id = points_g[0, -1].long().item()
                while (frame_id + tdir < num_frames) \
                        and (frame_id + tdir >= 0) \
                        and (visited[gid, frame_id+tdir]):
                    center = points_g.mean(0)[:3]
                    pose = transformations[gid, frame_id+tdir, :].cpu()
                    t, cost, sint = pose[:3], pose[3], pose[4]
                    
                    R2 = torch.tensor([[cost, -sint],
                                       [sint,  cost]])
                    points_g[:, :3] -= center
                    points_g[:, :2] = points_g[:, :2] @ R2.T
                    points_g[:, :3] = points_g[:, :3] + t + center
                    points_g[:, -1] += tdir
                    trace.append(points_g.cpu().clone())
                    frame_id += tdir
                if tdir == -1:
                    trace = trace[::-1]
            trace = torch.cat(trace, dim=0)
            if self.debug:
                self.vis.pointcloud(f'trace-{gid}', trace[:, :3])
                import ipdb; ipdb.set_trace()
                self.vis.show()
        if self.debug:
            self.vis.show()
        import ipdb; ipdb.set_trace()

    def motion_sync(self, voxels, voxels_velo, num_graphs, graph_idx, vp_edges, seq):
        points = torch.tensor(seq.points4d(), dtype=torch.float32)
        if self.debug:
            self.vis.pointcloud('points', points[:, :3].cpu())
        num_frames = len(seq.frames)
        ev, ep = vp_edges
        if self.velocity:
            points_velo = torch.zeros(points.shape[0], 3)
            points_velo[ep] = voxels_velo[ev]
        else:
            points_velo = torch.zeros(points.shape[0], 3)
        self.tracker.set_points(points)
        graph_idx_by_point = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                     dim=0, reduce='max',
                                     dim_size=points.shape[0])
        graph_idx_by_point, points, num_graphs = \
                self.filter_graphs(points, graph_idx_by_point,
                                   num_point_range=[5, 3000])
        if self.debug:
            data=torch.load(f'work_dirs/candidate_traces/seq_{seq.seq_id}.pt')
            pose = data['pose']
            visited = data['visited']
            selected_points = data['points']
            selected_graph_idx = data['graph_idx']
            num_selected_graphs = selected_graph_idx.long().max().item()+1
            self.visualize_trace(selected_graph_idx, selected_points, visited, pose,
                                 num_selected_graphs, num_frames)
        transformations = torch.zeros(num_graphs, num_frames, 5,
                                      dtype=torch.float64).cuda()
        trace_centers = torch.zeros(num_graphs, num_frames, 3, dtype=torch.float64).cuda()
        visited = torch.zeros(num_graphs, num_frames, dtype=torch.bool).cuda()
        self.track_dir(points, graph_idx_by_point, 1,
                       transformations, trace_centers, visited)
        self.track_dir(points, graph_idx_by_point, -1,
                       transformations, trace_centers, visited)

        idx0, idx1 = torch.where(visited)
        min_visited_frame_id = scatter(idx1, idx0, dim=0,
                                       dim_size=num_graphs, reduce='min')
        max_visited_frame_id = scatter(idx1, idx0, dim=0,
                                       dim_size=num_graphs, reduce='max')
        trace_length = max_visited_frame_id - min_visited_frame_id + 1
        center_l = trace_centers[(torch.arange(num_graphs), min_visited_frame_id)]
        center_r = trace_centers[(torch.arange(num_graphs), max_visited_frame_id)]
        travel_dist = (center_r - center_l).norm(p=2, dim=-1)
        mean_velo = travel_dist / trace_length
        
        is_active_graph = (trace_length >= 10).cpu() \
                          & (mean_velo > self.min_mean_velocity).cpu() \
                          & (travel_dist > self.min_travel_dist).cpu()
        save_path = f'work_dirs/candidate_traces/seq_{seq.seq_id}.pt'
        selected_points, selected_graph_idx = \
            self.translate(points, graph_idx_by_point, is_active_graph)
        save_dict = dict(
                        visited=visited[is_active_graph],
                        pose=transformations[is_active_graph],
                        trace_centers=trace_centers[is_active_graph],
                        graph_idx=selected_graph_idx,
                        points=selected_points,
                        points_all=points,
                    )
        torch.save(save_dict, save_path)

    def __call__(self, res, info):
        seq = res['lidar_sequence']
        if self.velocity:
            voxels, voxels_velo = res['voxels'], res['voxels_velo']
        else:
            voxels, voxels_velo = res['voxels'], torch.zeros(res['voxels'].shape[0], 3)
        ev, ep = res['vp_edges']
        num_graphs, graph_idx = res['num_graphs'], res['graph_idx']
        print(f'num_graphs={num_graphs}')
        total_time = time.time()
        self.motion_sync(voxels, voxels_velo, num_graphs,
                         graph_idx, res['vp_edges'], seq)
        print(f'elapsed time={time.time()-total_time}')
        
        return res, info 
