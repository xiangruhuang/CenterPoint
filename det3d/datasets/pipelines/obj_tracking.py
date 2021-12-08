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
from .kalman_filter import KalmanFilter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import glob

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
        self.kf_config = kf_config
        self.threshold = threshold
        self.acc_threshold = acc_threshold
        self.reg_threshold = reg_threshold
        self.angle_threshold = angle_threshold
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.corres_voxel_size = torch.tensor(corres_voxel_size,
                                              dtype=torch.float32)
        self.min_travel_dist=min_travel_dist
        self.min_mean_velocity=min_mean_velocity
        self.min_velo = min_velo
        self.velocity = velocity
        self.crop_points = crop_points
        self.debug=debug
        self.ht = HashTable(2000000)
        if self.debug:
            self.vis = Visualizer([], [])

    def register(self, points, frame, pc_range, temporal_offset):

        def solve(p, q):
            R = torch.eye(3).to(p.dtype).cuda()
            if p.shape[0] < 3:
                t = (q - p @ R.T).mean(0)
                error = (q - p @ R.T - t).norm(p=2, dim=-1).mean()
            else:
                for itr in range(1):
                    t = (q - p @ R.T).mean(0)
                    M = ( p[:, :2].unsqueeze(-1) @ (q-t)[:, :2].unsqueeze(-2) ).sum(0)
                    U, S, V = M.double().svd()
                    R2 = V @ U.T
                    if np.linalg.det(R2.cpu().numpy()) < 0:
                        R2 = V.clone()
                        R2[:, -1] *= -1
                        R2 = R2 @ U.T
                    R2 = R2.float()
                    R[:2, :2] = R2
                    error = (q - p @ R.T - t).norm(p=2, dim=-1).mean()
            return R, t, error

        voxel_size = self.corres_voxel_size.cuda()
        frame_id = points[0, -1].long().item() + temporal_offset
        for itr in range(3):
            if itr > 0:
                moving_points = points.clone()
                moving_points[:, :3] = moving_points[:, :3] @ R.T + t
            else:
                moving_points = points
            ep, ef = self.ht.find_corres(frame, moving_points, voxel_size,
                                         pc_range, temporal_offset).long()
            if ep.shape[0] * 2 < points.shape[0]:
                t = torch.zeros(3).float().cuda()
                R = torch.eye(3).float().cuda()
                error = 1e10
                break
            else:
                R, t, error = solve(points[ep, :3], frame[ef, :3])
        
        return R, t, error

    def check_status(self, centers, center, error):
        if error > self.threshold:
            if self.debug:
                print(f'too much registration error: {error:.4f}')
            return False

        if len(centers) > 1:
            v = center - centers[-1]
            v_last = centers[-1] - centers[-2]
            acc = (v_last - v).norm(p=2, dim=-1)
            if acc > self.acc_threshold:
                if self.debug:
                    print(f'too much acceleration: {acc:.4f}')
                return False

        return True

    def track_dir(self, _points, points_all, velocity,
                  frame_id, temporal_dir=1):
        """
        
        Returns:
            trace, centers, T, errors
        """
        pc_range = torch.cat([points_all.min(0)[0]-3,
                              points_all.max(0)[0]+3], dim=0).cuda()
        num_frames = points_all[:, -1].max().long()+1
        if temporal_dir == -1:
            end_frame = 0
        elif temporal_dir == 1:
            end_frame = num_frames-1
        else:
            raise ValueError('Temporal Dir needs to be in {-1, 1}')
        if frame_id == end_frame:
            return [], [], [], [], []
        points = _points.clone().cuda()
        shift = points.mean(0)[:3]
        points[:, :3] = points[:, :3] - shift 
        pc_range[:3] -= shift
        pc_range[4:7] -= shift
        x = torch.zeros(6)
        x[:3] = points.mean(0)[:3].cpu()
        x[3:] = velocity.mean(0).cpu() * temporal_dir
        center = x[:3]
        kf = KalmanFilter(x, **self.kf_config)
        num_points = points.shape[0]

        trace, centers, T, errors = [], [], [], []
        selected_indices = []
        while frame_id != end_frame:
            center_next = kf.predict()
            t0 = (center_next - center).cuda()
            points[:, :3] = points[:, :3] + t0
            frame_indices = torch.where(points_all[:, -1] \
                                        == (frame_id + temporal_dir))[0]
            frame = points_all[frame_indices].cuda()
            frame[:, :3] -= shift
            R, t, error = self.register(points, frame, pc_range, temporal_dir)
            points[:, :3] = points[:, :3] @ R.T + t
            points[:, -1] += temporal_dir
            center = points.mean(0)[:3].cpu()
            if not self.check_status(centers, center, error):
                break
            ep, ef = self.ht.voxel_graph(frame, points, self.voxel_size,
                                         0, radius=0.10,
                                         max_num_neighbors=128)

            selected_indices.append(frame_indices[ef.unique()])
            
            trace_this = points[:num_points].cpu()
            trace.append(trace_this)
            errors.append(error)
            centers.append(center)
            Ti = torch.eye(4)
            Ti[:3, :3] = R
            Ti[:3, 3] = t
            T.append(Ti)
            kf.update(center)
            frame_id = frame_id + temporal_dir

        for i in range(len(centers)):
            centers[i] += shift.cpu()
            trace[i][:, :3] += shift.cpu()

        if temporal_dir == -1:
            return trace[::-1], centers[::-1], T[::-1], errors[::-1], selected_indices[::-1]
        else:
            return trace, centers, T, errors, selected_indices

    def track(self, _points, points_all, _velocity):
        velocity = _velocity.clone().cuda()
        num_frames = points_all[:, -1].max().long()+1
        frame_id = _points[0, -1].long().item()

        # forward
        trace_f, centers_f, T_f, errors_f, selected_indices_f = \
            self.track_dir(_points, points_all, velocity, frame_id, 1)
        
        # backward
        frame_id = _points[0, -1].long().item()
        trace_b, centers_b, T_b, errors_b, selected_indices_b = \
            self.track_dir(_points, points_all, velocity, frame_id, -1)
        
        # merge
        trace = trace_b + [_points.cpu()] + trace_f
        centers = centers_b + [_points.mean(0)[:3].cpu()] + centers_f
        T = T_b + [torch.eye(4)] + T_f
        if (len(selected_indices_b) == 0) and \
                (len(selected_indices_f) == 0):
            selected_indices = torch.zeros(0).long()
        else:
            selected_indices = selected_indices_b + selected_indices_f
            selected_indices = torch.cat(selected_indices, dim=0).cpu()
        
        trace_stack = torch.stack(trace, dim=0)
        dr = torch.zeros_like(trace_stack)
        dr[:-1] += trace_stack[1:] - trace_stack[:-1]
        dr[1:] += trace_stack[1:] - trace_stack[:-1]
        dr[0] *= 2
        dr[-1] *= 2
        dr /= 2
        trace = [torch.cat([t, dr[i, :, :3]], dim=-1)
                    for i, t in enumerate(trace)]
        errors = errors_b + [torch.tensor(0.0)] + errors_f

        errors = torch.tensor(errors)
        T = torch.stack(T, dim=0)
        centers = torch.stack(centers, dim=0)
        return trace, centers, T, errors, selected_indices
        
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

    def update(self, points, points_gpu, trace, point_weight, point_dr):
        voxel_size = self.voxel_size
        et, ev = self.ht.voxel_graph(points_gpu,
                                     trace[:, :4].cuda(),
                                     voxel_size.cuda(),
                                     0, radius=1.0).cpu()
        dist = (trace[et, :3] - voxels[ev, :3]).norm(p=2, dim=-1)
        weight = np.exp(-(dist / 0.3)**2/2.0)

        weight_dr = weight[:, np.newaxis] * trace[et, 4:]
        ev_unique, ev_inv = ev.unique(return_inverse=True)
        weight_dense = scatter(weight, ev_inv, dim=0,
                               dim_size=ev_unique.shape[0], reduce='sum')
        voxel_weight[ev_unique] += weight_dense
        weight_dr_dense = scatter(weight_dr, ev_inv, dim=0,
                                  dim_size=ev_unique.shape[0],
                                  reduce='sum')
        voxel_dr[ev_unique] += weight_dr_dense

    def motion_sync(self, voxels, voxels_velo, num_graphs, graph_idx, vp_edges, seq):
        points = torch.tensor(seq.points4d(), dtype=torch.float32)
        if self.debug:
            vis = Visualizer([], [])
            ps_p = vis.pointcloud('points', points[:, :3])
            vis.boxes('box', seq.corners(), seq.classes())
        point_weight = torch.zeros(points.shape[0])
        point_dr = torch.zeros(points.shape[0], 3)
        points_velo = torch.zeros(points.shape[0], 3)
        ev, ep = vp_edges
        points_velo[ep] = voxels_velo[ev]
        mask = torch.zeros(points.shape[0], dtype=torch.bool)
        is_cropped = torch.zeros(points.shape[0], dtype=torch.bool)
        points_cropped = points.clone()
        original_indices = torch.arange(points.shape[0], dtype=torch.long)
        graph_idx_by_point = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                     dim=0, reduce='max',
                                     dim_size=points.shape[0])
        trace_count = 0

        for i in range(num_graphs):
            indices = torch.where((graph_idx_by_point == i) & (is_cropped == False))[0]
            if indices.shape[0] == 0:
                continue
            points_i = points[indices].clone()
            avg_velo = points_velo[indices].mean(0).norm(p=2)
            frame_id = points_i[0, -1].long()
            if (avg_velo > self.min_velo) and (points_i.shape[0] >= 10):
                if self.debug:
                    print(f'cluster {i}: ', end="")
                    ps_cluster = vis.pointcloud(f'cluster-{i}', points_i[:, :3], radius=3e-4, enabled=False)
                t0 = time.time()
                trace, centers, T, errors, selected_indices = \
                        self.track(points_i, points_cropped, points_velo)
                if len(trace) > 0:
                    print(f'average t={(time.time()-t0)/len(trace):.4f}')
                if not self.check_trace_status(trace, centers):
                    continue
                trace_i = torch.cat([points_i, points_cropped[selected_indices]],
                                     dim=0)
                trace = torch.cat(trace, dim=0)
                avg_time = (time.time()-t0)/(i+1)
                eta = avg_time * (num_graphs - i)
                print(f'pass {i:05d}, time={avg_time:.4f}, ETA={eta:.4f}, '\
                      f'num_points={points_cropped.shape[0]}')
                if self.crop_points: 
                    is_cropped[original_indices[selected_indices]] = True
                    mask[selected_indices] = True
                    original_indices = original_indices[mask[:points_cropped.shape[0]] == False]
                    points_cropped = points_cropped[mask[:points_cropped.shape[0]] == False]
                    mask[selected_indices] = False
                
                if self.debug:
                    ps_trace = vis.pointcloud(f'trace-{i}', trace[:, :3], radius=3e-4, enabled=False)
                    ps_trace.add_scalar_quantity('frame', trace[:, -1])
                    ps_c = vis.trace(f'center-trace-{i}', centers[:, :3], enabled=False, radius=3e-4)
                    ps_c.add_scalar_quantity('error', errors.cpu(), defined_on='nodes')
                    vis.pointcloud(f'selected-trace-{i}', trace_i[:, :3], radius=3e-4)
                    ps_p = vis.pointcloud('points', points_cropped[:, :3])

                trace_dict = dict(cls={}, box={}, T={}, points={})
                if self.debug:
                    box_corners, box_classes, points_in_box = [], [], [] 

                for frame_id in range(trace_i[:, -1].min().long(),
                                      trace_i[:, -1].max().long()+1):
                    mask_f = (trace_i[:, -1] == frame_id)
                    center_f = trace_i[mask_f].mean(0)[:3]
                    T_f = seq.frames[frame_id].pose
                    trace_dict['points'][frame_id] = trace_i[mask_f]
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
                            mask_f = points_cropped[:, -1] == frame_id
                            points_f = points_cropped[mask_f]
                            surfaces = box_np_ops.corner_to_surfaces_3d(box_corners_f[box_id][np.newaxis, ...])
                            indices = points_in_convex_polygon_3d_jit(points_f.numpy()[:, :3], surfaces)[:, 0]
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
        self.motion_sync(voxels, voxels_velo, num_graphs,
                         graph_idx, res['vp_edges'], seq)
        
        return res, info 
