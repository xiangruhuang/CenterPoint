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

    def track(self, cluster, cluster_velo, points, seq):
        if self.debug:
            vis = Visualizer([], [])
            ps_cluster = vis.pointcloud(f'cluster', cluster[:, :3],
                                        radius=3e-4,
                                        enabled=False)
            vis.show()
        trace, centers, T, errors, selected_indices = \
                self.tracker.track(cluster, cluster_velo)
        #if len(trace) > 0:
        #    print(f'average t={(time.time()-t0)/len(trace):.4f}')
        if not self.check_trace_status(trace, centers):
            return None, None 
        trace_this = torch.cat([cluster, points[selected_indices].cpu()],
                               dim=0)
        trace = torch.cat(trace, dim=0)
        
        if self.debug:
            ps_trace = vis.pointcloud(f'trace', trace[:, :3],
                                      radius=3e-4, enabled=False)
            ps_trace.add_scalar_quantity('frame', trace[:, -1])
            ps_c = vis.trace(f'center-trace', centers[:, :3],
                             enabled=False, radius=3e-4)
            ps_c.add_scalar_quantity('error', errors.cpu(), defined_on='nodes')
            vis.pointcloud(f'selected-trace', trace_this[:, :3], radius=3e-4)
            ps_p = vis.pointcloud('points', points[:, :3].cpu())
        
        trace_dict = dict(cls={}, box={}, T={}, points={},
                          errors=errors)
        if self.debug:
            box_corners, box_classes, points_in_box = [], [], []

        for frame_id in range(trace_this[:, -1].min().long(),
                              trace_this[:, -1].max().long()+1):
            mask_f = (trace_this[:, -1] == frame_id)
            center_f = trace_this[mask_f].mean(0)[:3]
            T_f = seq.frames[frame_id].pose
            trace_dict['points'][frame_id] = trace_this[mask_f]
            trace_dict['T'][frame_id] = T_f
            box_corners_f = seq.corners(frame_id, frame_id+1)
            box_classes_f = seq.classes(frame_id, frame_id+1)
            box_centers_f = box_corners_f.mean(1)
            box_centers_f = torch.tensor(box_centers_f,
                                         dtype=torch.float32)
            dist = (box_centers_f - center_f).norm(p=2, dim=-1)
            if dist.shape[0] > 0:
                box_id = dist.argmin()
                if self.debug:
                    from det3d.core.bbox import box_np_ops
                    from det3d.core.bbox.geometry import (
                        points_count_convex_polygon_3d_jit,
                        points_in_convex_polygon_3d_jit,
                    )
                    box_corners.append(torch.tensor(box_corners_f[box_id]))
                    box_classes.append(torch.tensor(box_classes_f[box_id]))
                    mask_f = points[:, -1] == frame_id
                    points_f = points[mask_f]
                    surfaces = box_np_ops.corner_to_surfaces_3d(
                                   box_corners_f[box_id][np.newaxis, ...])
                    indices = points_in_convex_polygon_3d_jit(
                                  points_f.numpy()[:, :3], surfaces)[:, 0]
                    points_in_box.append(points_f[indices])

                trace_dict['box'][frame_id] = seq.frames[frame_id].boxes[box_id]
                trace_dict['cls'][frame_id] = seq.frames[frame_id].classes[box_id]

        if self.debug:
            box_corners = torch.stack(box_corners, dim=0)
            box_classes = torch.stack(box_classes, dim=0)
            vis.boxes('selected box', box_corners, box_classes)
            points_in_box = torch.cat(points_in_box, dim=0)
            vis.pointcloud('points in box', points_in_box[:, :3])
            import ipdb; ipdb.set_trace()
            vis.show()
        return trace_dict, selected_indices

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
                  transformations, visited):
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            vis.pointcloud('points', points[:, :3].cpu())
            vis.show()

        points = points.clone().double().cuda()
        graph_idx = graph_idx.clone().cuda()
        num_graphs = graph_idx.max().item() + 1
        graph_size = scatter(torch.ones_like(graph_idx), graph_idx,
                             dim=0, dim_size=num_graphs, reduce='sum')
        original_graph_indices = torch.arange(num_graphs, dtype=torch.long).cuda()
        centers = scatter(points[:, :3], graph_idx, dim=0,
                          dim_size=num_graphs, reduce='mean')
        last_centers = None

        while True:
            import ipdb; ipdb.set_trace()
            gwise_R, gwise_t, gwise_error = \
                                self.tracker.track_graphs(
                                    points, graph_idx, graph_size, 1)

            # update transformations
            gwise_theta = gwise_R[:, 0, 0].clip(-1, 1).arccos()
            gwise_pose = torch.cat([gwise_t, gwise_theta.unsqueeze(-1)], dim=-1)
            gwise_frame_id = scatter(points[:, -1], graph_idx, dim=0,
                                     dim_size=original_graph_indices.shape[0],
                                     reduce='max').long()

            is_active_graph = gwise_error < self.reg_threshold
            if is_active_graph.any() == False:
                break
            is_active_point = is_active_graph[graph_idx]
            pwise_t, pwise_R = gwise_t[graph_idx], gwise_R[graph_idx]
            if self.debug:
                ps_c = vis.pointcloud('graph_centers',
                                      centers[original_graph_indices, :3].cpu())
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0, dim_size=num_graphs,
                                    reduce='mean')
            points[:, :3] -= gwise_centers[graph_idx]
            points[:, :3] = (pwise_R @ points[:, :3].unsqueeze(-1)).squeeze(-1) + pwise_t
            points[:, :3] += gwise_centers[graph_idx]
            gwise_centers = scatter(points[:, :3], graph_idx, dim=0,
                                    dim_size=original_graph_indices.shape[0],
                                    reduce='mean')
            if self.debug:
                ps_c.add_vector_quantity('velocity',
                                         (gwise_centers \
                                          - centers[original_graph_indices]).cpu())
                vis.show()

            points, graph_idx = self.translate(points, graph_idx, is_active_graph)
            points[:, -1] += temporal_dir
            
            # check acceleration
            if last_centers is not None:
                new_centers = scatter(points, graph_idx, dim=0,
                                      dim_size=original_graph_indices.shape[0],
                                      reduce='mean')
                velocity = new_centers - centers[original_graph_indices]
                last_velocity = centers[original_graph_indices] \
                                - last_centers[original_graph_indices]
                acc = (last_velocity - velocity).norm(p=2, dim=-1)
                is_active_graph = (acc > self.acc_threshold) & is_active_graph
                if is_active_graph.any() == False:
                    break

            # update graph indices for next iteration
            original_graph_indices = original_graph_indices[is_active_graph]

            # update transformations and visited
            gwise_frame_id = gwise_frame_id[is_active_graph]
            gwise_pose = gwise_pose[is_active_graph]
            visited[(original_graph_indices, gwise_frame_id)] = True
            transformations[(original_graph_indices, gwise_frame_id)] = gwise_pose

            # update last centers
            last_centers = centers.clone()
            centers[original_graph_indices] = \
                    scatter(points, graph_idx, dim=0,
                            dim_size=original_graph_indices.shape[0],
                            reduce='mean')

    def motion_sync(self, voxels, voxels_velo, num_graphs, graph_idx, vp_edges, seq):
        points = torch.tensor(seq.points4d(), dtype=torch.float32)
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
        transformations = torch.zeros(num_graphs, num_frames, 4,
                                      dtype=torch.float64).cuda()
        visited = torch.zeros(num_graphs, num_frames, dtype=torch.bool).cuda()
        self.track_dir(points, graph_idx_by_point, 1,
                       transformations, visited)
        self.track_dir(points, graph_idx_by_point, -1,
                       transformations, visited)

        visited = torch.zeros(points.shape[0]).long()
        sorted_indices = torch.argsort(deg, descending=True)
        print(f'num graphs = {mask.sum()}')
        trace_count = 0

        start_time = time.time()
        for i, g_index in enumerate(sorted_indices[mask]):
            indices = torch.where(graph_idx_by_point == g_index)[0]
            #print(f'cluster {i}: num points={indices.shape[0]}')
            points_i = points[indices].clone()
            points_velo_i = points_velo[indices]
            avg_velo = points_velo_i.mean(0).norm(p=2)
            frame_id = points_i[0, -1].long()
            visited_i = visited[indices]
            visited_ratio = visited_i.sum() / indices.shape[0]
            if (avg_velo > self.min_velo):
                if (visited_i.sum() > 10) or (visited_ratio > 0.7):
                    print(f'visited = {visited}, ratio = {visited_ratio}')
                    avg_time = (time.time()-start_time) / (i+1)
                    ETA = (mask.sum() - i) * avg_time
                    print(f'visited {g_index}, ETA={ETA}')
                    continue
                trace_dict, selected_indices = \
                    self.track(points_i, points_velo_i, points, seq)
                if selected_indices is not None:
                    visited[selected_indices.cpu()] = 1
                if trace_dict is None:
                    avg_time = (time.time()-start_time) / (i+1)
                    ETA = (mask.sum() - i) * avg_time
                    print(f'failed {g_index}, ETA={ETA}')
                    continue
                avg_time = (time.time()-start_time) / (i+1)
                ETA = (mask.sum() - i) * avg_time
                print(f'pass {g_index}, ETA={ETA}')
                save_path = os.path.join(
                                'work_dirs',
                                'object_traces',
                                f'seq_{seq.seq_id}_trace_{trace_count}.pt')
                torch.save(trace_dict, save_path)
                trace_count += 1

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
