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
from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import (
    points_in_convex_polygon_3d_jit,
)
from .object_trace_compare import ObjectTraceCompare

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
                  transformations, trace_centers, visited, errors):
        points = points.clone().double().cuda()
        num_frames = points[:, -1].long().max().item() + 1
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
            #points[:, :3] -= gwise_centers[graph_idx]
            points[:, :3] = (pwise_R @ points[:, :3].unsqueeze(-1)).squeeze(-1) + pwise_t
            #points[:, :3] += gwise_centers[graph_idx]
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
            
            points, graph_idx = self.translate(points, graph_idx,
                                               is_active_graph)

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
            theta = gwise_R[:, 0, 0].clip(-1, 1).arccos() * gwise_R[:, 0, 0].sign()
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

            # update registration error
            gwise_error = gwise_error[is_active_graph]
            errors[(original_graph_indices, gwise_frame_id + temporal_dir)] = gwise_error

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
            ps_p = self.vis.pointcloud('points', points[:, :3].cpu()+seq.scene_center)
            ps_p.add_scalar_quantity('frame', points[:, -1].cpu())
        num_frames = len(seq.frames)
        ev, ep = vp_edges
        if self.velocity:
            points_velo = torch.zeros(points.shape[0], 3)
            points_velo[ep] = voxels_velo[ev]
        else:
            points_velo = torch.zeros(points.shape[0], 3)
        self.tracker.set_points(points)
        self.compare = ObjectTraceCompare(points.shape[0])
        graph_idx_by_point_ori = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                         dim=0, reduce='max',
                                         dim_size=points.shape[0])
        if True:
            graph_idx_by_point, points, num_graphs = \
                    self.filter_graphs(points, graph_idx_by_point_ori,
                                       num_point_range=[5, 3000])
            transformations = torch.zeros(num_graphs, num_frames, 4,
                                          dtype=torch.float64).cuda()
            trace_centers = torch.zeros(num_graphs, num_frames, 3,
                                        dtype=torch.float64).cuda()
            errors = torch.zeros(num_graphs, num_frames, dtype=torch.float64).cuda()
            visited = torch.zeros(num_graphs, num_frames, dtype=torch.bool).cuda()
            self.track_dir(points, graph_idx_by_point, 1,
                           transformations, trace_centers, visited, errors)
            self.track_dir(points, graph_idx_by_point, -1,
                           transformations, trace_centers, visited, errors)
            if self.debug:
                save_dict = {}
                save_dict['transformations'] = transformations
                save_dict['visited'] = visited
                save_dict['trace_centers'] = trace_centers
                save_dict['graph_idx_by_point'] = graph_idx_by_point
                save_dict['points'] = points
                save_dict['num_graphs'] = num_graphs
                save_dict['errors'] = errors
                torch.save(save_dict, 'saved_tracking.pt')
        else:
            save_dict = torch.load('saved_tracking.pt')
            transformations = save_dict['transformations']
            visited = save_dict['visited']
            graph_idx_by_point = save_dict['graph_idx_by_point']
            points = save_dict['points'].cpu()
            num_graphs = save_dict['num_graphs']
            errors = save_dict['errors']
            trace_centers = save_dict['trace_centers']
        
        idx0, idx1 = torch.where(visited)
        min_visited_frame_id = scatter(idx1, idx0, dim=0,
                                       dim_size=num_graphs, reduce='min')
        max_visited_frame_id = scatter(idx1, idx0, dim=0,
                                       dim_size=num_graphs, reduce='max')
        trace_length = max_visited_frame_id - min_visited_frame_id + 1
        center_l = trace_centers[(torch.arange(num_graphs), min_visited_frame_id)]
        center_r = trace_centers[(torch.arange(num_graphs), max_visited_frame_id)]
        travel_dist = (center_r - center_l)[:, :3].norm(p=2, dim=-1)
        mean_velo = travel_dist / trace_length
        
        is_active_graph = (trace_length >= 10) \
                          & (mean_velo > self.min_mean_velocity) \
                          & (travel_dist > self.min_travel_dist)
        if is_active_graph.sum() < 0.5:
            print(f'num active graph={is_active_graph.sum()}')
            save_path = f'work_dirs/candidate_traces/seq_{seq.seq_id}.pt'
            print(f'saving to {save_path}')
            torch.save([], save_path)
            return
        
        points = points.cuda()
        graph_idx_by_point = graph_idx_by_point.cuda()
        selected_points, selected_graph_idx = \
            self.translate(points, graph_idx_by_point, is_active_graph)
        trace_length = trace_length[is_active_graph]
        travel_dist = travel_dist[is_active_graph]
        num_active_graphs = selected_graph_idx.long().max().item() + 1
        visited = visited[is_active_graph]
        transformations=transformations[is_active_graph]
        trace_centers=trace_centers[is_active_graph]
        errors = errors[is_active_graph]

        print(f'num active graph={num_active_graphs}')
        traces = []
        box_centers = torch.tensor(seq.box_centers_4d()).float()
        box_geom_centers = torch.zeros(box_centers.shape[0], 4).float()
        box_geom_centers[:, :3] = box_centers[:, :3]
        corners = torch.tensor(seq.corners())
        classes = torch.tensor(seq.classes())
        boxes = torch.tensor(seq.boxes())
        scene_center = torch.tensor(seq.scene_center)
        points_in_box_dict = {}
        for fid in range(num_frames):
            print(f'frame={fid}')
            mask = self.tracker.points[:, -1].long() == fid
            box_ids = torch.where(box_centers[:, -1].long() == fid)[0]
            if box_ids.shape[0] == 0:
                continue
            surfaces = box_np_ops.corner_to_surfaces_3d(
                           corners[box_ids].cpu().numpy()
                       )
            frame = self.tracker.points[mask, :3].cpu().numpy()
            indices = points_in_convex_polygon_3d_jit(
                          frame, surfaces)
            for i, b in enumerate(box_ids):
                mask_b = indices[:, i]
                if mask_b.sum() > 0.5:
                    box_geom_centers[b.item(), :3] = torch.tensor(frame[mask_b].mean(0))
                points_in_box_dict[b.item()] = frame[mask_b]

        box_centers[:, -1] *= 10000
        box_geom_centers[:, -1] = box_centers[:, -1]
        graph_list = sorted(torch.arange(num_active_graphs),
                            key = lambda i: trace_length[i],
                            reverse=True)

        for i, gid in enumerate(graph_list):
            # find points near the trace
            mask_g = selected_graph_idx == gid
            points_g = selected_points[mask_g].clone().double()
            cluster_g = points_g.clone()
            act_frame_ids = torch.where(visited[gid, :])[0]
            num_act_frames = act_frame_ids.shape[0]
            num_act_points = mask_g.long().sum().item()
            transformations_g = transformations[gid]
            pose_g = transformations[gid, act_frame_ids]
            t_g, theta_g = pose_g[:, :3], pose_g[:, 3]
            R_g = torch.stack([theta_g.cos(), -theta_g.sin(),
                               theta_g.sin(),  theta_g.cos()],
                              dim=-1).view(-1, 2, 2)
            errors_g = errors[gid, act_frame_ids]
            points_g = points_g.repeat(num_act_frames, 1, 1)
            points_g = points_g
            points_g[:, :, :2] = points_g[:, :, :2] @ R_g.transpose(1, 2)
            points_g[:, :, :3] += t_g.unsqueeze(-2)
            points_g[:, :, 3] = act_frame_ids.unsqueeze(-1)
            points_g = points_g.view(-1, 4).float()
            selected_indices = self.tracker.ht.points_in_radius_step2(
                                   points_g, 0, 0.5,
                               )
            #eq, er = self.tracker.ht.voxel_graph_step2(points_g, 0, 2.5, 256)
            #mask_dist = (points_g[eq, :2] - self.tracker.points[er, :2]
            #             ).norm(p=2, dim=-1) < 1.0
            #eq, er = eq[mask_dist], er[mask_dist]
            #selected_indices = er.unique()

            # find nearest ground truth boxes
            selected_nbrs = self.tracker.points[selected_indices].cpu()
            frame_ids = selected_nbrs[:, -1].long()
            selected_centers = scatter(selected_nbrs, frame_ids, dim=0,
                                       dim_size=num_frames, reduce='sum')
            num_frames_ = scatter(torch.ones_like(frame_ids), frame_ids, dim=0,
                                  dim_size=num_frames, reduce='sum')
            mask = num_frames_ > 0
            selected_centers = selected_centers[mask] / num_frames_[mask].unsqueeze(-1)
            selected_centers[:, -1] *= 10000
            _, box_ids = knn(box_geom_centers, selected_centers, 1)
            dist = (selected_centers - box_geom_centers[box_ids]).norm(p=2, dim=-1)

            # find all points in ground truth boxes
            points_in_box = []
            for b in box_ids:
                points_in_box_b = torch.tensor(points_in_box_dict[b.item()])
                points_in_box.append(points_in_box_b)
            if len(points_in_box) > 0:
                points_in_box = torch.cat(points_in_box, dim=0)
            else:
                print('empty')
                points_in_box = torch.zeros(0, 3, dtype=torch.float32).to(scene_center)
            
            # shift to original coordinate system
            selected_nbrs[:, :3] += scene_center
            
            # add trace to queue
            survive, conflict_list = self.compare.compare(selected_indices.cpu())
            print(f'working on {i}, num trace = {len(self.compare.trace_dict.keys())}')
            if survive:
                # construct trace dictionary
                trace_dict = dict(
                                  points=selected_nbrs.cpu(),
                                  cluster=cluster_g,
                                  point_indices=selected_indices.cpu(),
                                  boxes=boxes[box_ids].cpu(),
                                  box_frame_ids=box_centers[box_ids, -1].cpu()/10000.0,
                                  corners=(corners[box_ids]+scene_center).cpu(),
                                  classes=classes[box_ids].cpu(),
                                  points_in_box=points_in_box.cpu()+scene_center,
                                  errors=errors_g,
                                  transformations=pose_g,
                                  dist=dist.mean(),
                                 )
                self.compare.add_object_trace(
                    gid, trace_dict, conflict_list
                )
                if self.debug:
                    self.vis.pointcloud(f'cluster-{gid}', 
                                        points_g[:, :3].cpu()+scene_center,
                                        enabled=False)
                    corners_this = trace_dict['corners']
                    classes_this = trace_dict['classes']
                    self.vis.boxes(f'box-{i}', corners_this,
                                   classes_this, enabled=False)
                    points = trace_dict['points']
                    ps_t = self.vis.pointcloud(f'trace-{i}', points[:, :3],
                                               radius=2.1e-4)
                    z_axis = torch.zeros(points.shape[0], 3)
                    z_axis[:, -1] = 1.0
                    ps_t.add_vector_quantity('z-axis', z_axis)
                    points_in_box = trace_dict['points_in_box']
                    self.vis.pointcloud(f'points-in-box-{i}',
                                        points_in_box[:, :3].cpu(),
                                        radius=2.05e-4, enabled=False)
                    self.vis.pointcloud(f'moving-frame-{i}',
                                        points_g[:, :3].cpu()+scene_center,
                                        radius=2.1e-4, enabled=False)
                    self.vis.show()
                    import ipdb; ipdb.set_trace()

        save_path = f'work_dirs/candidate_traces/seq_{seq.seq_id}.pt'
        print(f'saving to {save_path}')
        for trace_id, trace_dict in self.compare.trace_dict.items():
            if self.debug:
                corners = trace_dict['corners']
                classes = trace_dict['classes']
                self.vis.boxes(f'box-{trace_id}', corners,
                               classes, enabled=False)
                points = trace_dict['points']
                ps_t = self.vis.pointcloud(f'trace-{trace_id}', points[:, :3],
                                           radius=2.1e-4)
                z_axis = torch.zeros(points.shape[0], 3)
                z_axis[:, -1] = 1.0
                ps_t.add_vector_quantity('z-axis', z_axis)
                points_in_box = trace_dict['points_in_box']
                self.vis.pointcloud(f'points-in-box-{gid}',
                                    points_in_box[:, :3].cpu(),
                                    radius=2.05e-4, enabled=False)
                #self.vis.pointcloud(f'moving-frame-{gid}',
                #                    points_g[:, :3].cpu(),
                #                    radius=2.1e-4, enabled=False)
                #if (trace_id + 1) % 10 == 0:
                #    import ipdb; ipdb.set_trace()
                #    self.vis.show()
            traces.append(trace_dict)
            
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
