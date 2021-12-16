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
from torch_cluster import knn

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
        num_graphs = mask.long().sum().item()

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

            is_active_graph = gwise_error < self.reg_threshold
            if is_active_graph.any() == False:
                break
            is_active_point = is_active_graph[graph_idx]
            pwise_t, pwise_R = gwise_t[graph_idx], gwise_R[graph_idx]
            #if self.debug:
            #    ps_c = self.vis.pointcloud('graph_centers',
            #                          centers[original_graph_indices, :3].cpu())
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=num_existing_graphs,
                                    reduce='mean')
            points[:, :3] -= gwise_centers[graph_idx]
            points[:, :3] = (pwise_R @ points[:, :3].unsqueeze(-1)).squeeze(-1) + pwise_t
            points[:, :3] += gwise_centers[graph_idx]
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=num_existing_graphs,
                                    reduce='mean')
            #if self.debug:
            #    ps_c.add_vector_quantity('velocity',
            #                             (gwise_centers \
            #                              - centers[original_graph_indices]).cpu())

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

            # update graph indices and frame id
            original_graph_indices = original_graph_indices[is_active_graph]
            num_existing_graphs = original_graph_indices.shape[0]
            gwise_frame_id = scatter(points[:, -1], graph_idx, dim=0,
                                     dim_size=num_existing_graphs,
                                     reduce='max').long()

            # 1. update global transformations
            gwise_R = gwise_R[is_active_graph]
            gwise_t = gwise_t[is_active_graph]
            # 1.a) compute local rotation angle in [-pi, pi]
            theta = gwise_R[:, 0, 0].arccos() * gwise_R[:, 0, 0].sign()
            # 1.b) compute global rotation angle
            global_theta = transformations[(original_graph_indices,
                                            gwise_frame_id)][:, 3].clone()+theta
            # 1.c) compute global translation
            global_t = transformations[(original_graph_indices,
                                        gwise_frame_id)][:, :3].clone()
            global_t = (gwise_R @ global_t.unsqueeze(-1)).squeeze(-1) + gwise_t
            global_pose = torch.cat([global_t, global_theta.unsqueeze(-1)],
                                    dim=-1)
            # 1.d) update 
            transformations[(original_graph_indices, 
                             gwise_frame_id + temporal_dir)] = global_pose

            # update visited and centers
            visited[(original_graph_indices, 
                     gwise_frame_id+temporal_dir)] = True
            trace_centers[(original_graph_indices,
                           gwise_frame_id+temporal_dir)] = \
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
            
            # moving to next iteration
            points[:, -1] += temporal_dir

    def visualize_trace(self, graph_idx, points, visited, transformations,
                        num_selected_graphs, num_frames):
        if self.debug:
            print(f'num trace = {num_selected_graphs}')
        for gid in range(num_selected_graphs):
            mask_g = graph_idx == gid
            trace = [points[mask_g].cpu().clone()]
            size = (points[mask_g].max(0)[0] - points[mask_g].min(0)[0])[:2].max().item()
            for tdir in [-1, 1]:
                points_g = points[mask_g].cpu().clone().double()
                frame_id = points_g[0, -1].long().item()
                while (frame_id + tdir < num_frames) \
                        and (frame_id + tdir >= 0) \
                        and (visited[gid, frame_id+tdir]):
                    pose = transformations[gid, frame_id+tdir, :].cpu()
                    t, theta = pose[:3], pose[3], pose[4]
                    
                    R2 = torch.tensor([[theta.cos(), -theta.sin()],
                                       [theta.sin(),  theta.cos()]])
                    points_g[:, :2] = points_g[:, :2] @ R2.T
                    points_g[:, :3] = points_g[:, :3] + t
                    points_g[:, -1] += tdir
                    trace.append(points_g.cpu().clone())
                    frame_id += tdir
                if tdir == -1:
                    trace = trace[::-1]
            trace = torch.cat(trace, dim=0)
            if self.debug:
                self.vis.pointcloud(f'trace-{gid}', trace[:, :3], enabled=(size < 10))
                if (gid + 1) % 100 == 0:
                    import ipdb; ipdb.set_trace()
                    self.vis.show()
        if self.debug:
            self.vis.show()
        import ipdb; ipdb.set_trace()

    def motion_sync(self, voxels, voxels_velo, num_graphs, graph_idx, vp_edges, seq):
        seq.center()
        points = torch.tensor(seq.points4d(), dtype=torch.float32)
        if self.debug:
            self.vis.pointcloud('points', points[:, :3].cpu()+seq.scene_center)
        num_frames = len(seq.frames)
        ev, ep = vp_edges
        if self.velocity:
            points_velo = torch.zeros(points.shape[0], 3)
            points_velo[ep] = voxels_velo[ev]
        else:
            points_velo = torch.zeros(points.shape[0], 3)
        self.tracker.set_points(points)
        graph_idx_by_point_ori = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                         dim=0, reduce='max',
                                         dim_size=points.shape[0])
        graph_idx_by_point, points, num_graphs = \
                self.filter_graphs(points, graph_idx_by_point_ori,
                                   num_point_range=[5, 3000])
        transformations = torch.zeros(num_graphs, num_frames, 4,
                                      dtype=torch.float64).cuda()
        trace_centers = torch.zeros(num_graphs, num_frames, 3,
                                    dtype=torch.float64).cuda()
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
        
        is_active_graph = (trace_length >= 10) \
                          & (mean_velo > self.min_mean_velocity) \
                          & (travel_dist > self.min_travel_dist)
        points = points.cuda()
        graph_idx_by_point = graph_idx_by_point.cuda()
        selected_points, selected_graph_idx = \
            self.translate(points, graph_idx_by_point, is_active_graph)
        num_active_graphs = selected_graph_idx.long().max().item() + 1
        visited = visited[is_active_graph]
        transformations=transformations[is_active_graph]
        trace_centers=trace_centers[is_active_graph]

        print(f'num active graph={num_active_graphs}')
        traces = []
        box_centers = torch.tensor(seq.box_centers_4d()).float()
        box_centers[:, -1] *= 10000
        corners = torch.tensor(seq.corners())
        classes = torch.tensor(seq.classes())
        boxes = torch.tensor(seq.boxes())
        scene_center = torch.tensor(seq.scene_center)
        for gid in range(num_active_graphs):
            mask_g = selected_graph_idx == gid
            points_g = selected_points[mask_g].clone().double()
            act_frame_ids = torch.where(visited[gid, :])[0]
            num_act_frames = act_frame_ids.shape[0]
            num_act_points = mask_g.long().sum().item()
            pose_g = transformations[gid, act_frame_ids]
            t_g, theta_g = pose_g[:, :3], pose_g[:, 3]
            R_g = torch.stack([theta_g.cos(), -theta_g.sin(),
                               theta_g.sin(),  theta_g.cos()],
                              dim=-1).view(-1, 2, 2)
            points_g = points_g.repeat(num_act_frames, 1, 1)
            points_g[:, :, :2] = points_g[:, :, :2] @ R_g.transpose(1, 2)
            points_g[:, :, :3] += t_g.unsqueeze(-2)
            points_g[:, :, 3] = act_frame_ids.unsqueeze(-1)
            points_g = points_g.view(-1, 4).float()
            eq, er = self.tracker.ht.voxel_graph_step2(points_g, 0, 1.0, 64)
            dist = (points_g[eq, :2] - self.tracker.points[er, :2]
                    ).norm(p=2, dim=-1)
            eq, er = eq[dist < 0.1], er[dist < 0.1]
            selected_nbrs = self.tracker.points[er].cpu()
            frame_ids = selected_nbrs[:, -1].long()
            selected_centers = scatter(selected_nbrs, frame_ids, dim=0,
                                       dim_size=num_frames, reduce='sum')
            num_frames_ = scatter(torch.ones_like(frame_ids), frame_ids, dim=0,
                                  dim_size=num_frames, reduce='sum')
            mask = num_frames_ > 0
            selected_centers = selected_centers[mask] / num_frames_[mask].unsqueeze(-1)
            selected_centers[:, -1] *= 10000
            _, box_ids = knn(box_centers, selected_centers, 1)
            
            # shift to original coordinate system
            selected_nbrs[:, :3] += scene_center
            trace_dict = dict(
                              points=selected_nbrs,
                              boxes=boxes[box_ids],
                              corners=corners[box_ids]+scene_center,
                              classes=classes[box_ids],
                             )

            if self.debug:
                self.vis.boxes(f'box-{gid}', corners[box_ids]+scene_center, classes[box_ids])
                self.vis.pointcloud(f'trace-{gid}', selected_nbrs[:, :3])
                if (gid + 1) % 10 == 0:
                    import ipdb; ipdb.set_trace()
                    self.vis.show()
            traces.append(trace_dict)
        save_path = f'work_dirs/candidate_traces/seq_{seq.seq_id}.pt'
        print(f'saving to {save_path}')
        torch.save(traces, save_path)

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
