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
from .kalman_filter import KalmanFilter

class KalmanTracker(object):
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

        # temporary variables
        self.R = torch.eye(3).double().cuda()
        self.t = torch.zeros(3).double().cuda()

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
    
    def register(self, points):

        def solve(p, q):
            q = q.double()
            R = self.R
            R[:] = 0; R[0, 0] = 1; R[1, 1] = 1; R[2, 2] = 1;
            if p.shape[0] < 3:
                t = (q - p @ R.T).mean(0)
                error = (q - p @ R.T - t).norm(p=2, dim=-1).mean()
            else:
                for itr in range(2):
                    t = (q - p @ R.T).mean(0)
                    M = ( p[:, :2].unsqueeze(-1) @ (q-t)[:, :2].unsqueeze(-2) ).sum(0)
                    U, S, VT = np.linalg.svd(M.cpu().double())
                    V = VT.T
                    R2 = V @ U.T
                    if np.linalg.det(R2) < 0:
                        V[:, -1] *= -1
                        R2 = V @ U.T
                    R[:2, :2] = torch.tensor(R2).cuda()
                    error = (q - p @ R.T - t).norm(p=2, dim=-1).mean()
            return R, t, error

        points = points.double()
        #R = torch.eye(3, dtype=torch.float64).cuda()
        #t = torch.zeros(3, dtype=torch.float64).cuda()
        R = self.R
        R[:] = 0; R[0, 0] = 1; R[1, 1] = 1; R[2, 2] = 1;
        t = self.t
        t[:] = 0.0
        for itr in range(3):
            moving_points = points.clone()
            moving_points[:, :3] = moving_points[:, :3] @ R.T + t
            try:
                #ep, ef = self.ht.find_corres(frame, moving_points,
                #                             self.corres_voxel_size,
                #                             pc_range, 0).long()
                st=time.time()
                ep, ef = self.ht.find_corres_step2(moving_points.float(),
                                                   0).long()
                #print(f'find corres: npoints={moving_points.shape[0]}, time={time.time()-st}')
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
                print(e)
            if ep.shape[0] * 2 < points.shape[0]:
                #R = torch.eye(3, dtype=torch.float64).cuda()
                #t = torch.zeros(3, dtype=torch.float64).cuda()
                R[:] = 0; R[0, 0] = 1; R[1, 1] = 1; R[2, 2] = 1;
                t[:] = 0.0
                error = 1e10
                break
            else:
                st=time.time()
                R, t, error = solve(points[ep, :3], self.points[ef, :3])
                #print(f'solve time={time.time()-st}')
        
        return R, t, error
    
    def track_dir(self, _cluster, velocity,
                  frame_id, end_frame, temporal_dir):
        if frame_id == end_frame:
            return [], [], [], [], []
        cluster = _cluster.double().clone().cuda()
        pc_range = self.pc_range.clone()
        x = torch.zeros(6, dtype=torch.float64)
        x[:3] = cluster.mean(0)[:3].cpu()
        x[3:] = velocity.mean(0).cpu() * temporal_dir
        center = x[:3]
        kf = KalmanFilter(x, **self.kf_config)

        trace, centers, T, errors = [], [], [], []
        selected_indices = []
        while frame_id != end_frame:
            center_next = kf.predict()
            t0 = (center_next - center).to(cluster.device)
            cluster[:, :3] = cluster[:, :3] + t0
            cluster[:, -1] += temporal_dir
            start_time = time.time()
            R, t, error = self.register(cluster)
            #print(f'reg time={time.time()-start_time}')
            cluster[:, :3] = cluster[:, :3] @ R.T + t
            center = cluster.mean(0)[:3].cpu()
            if not self.check_status(centers, center, error):
                break
            ec, ep = self.ht.voxel_graph_step2(
                         cluster.float(), 0,
                         radius=1.0, max_num_neighbors=128)
            dist = (cluster[ec, :2] - self.points[ep, :2]).norm(p=2, dim=-1)
            mask = dist < 0.1
            ep = ep[mask]

            selected_indices.append(ep.unique())
            
            trace.append(cluster.cpu())
            errors.append(error)
            centers.append(center)
            Ti = torch.eye(4); Ti[:3, :3] = R.cpu(); Ti[:3, 3] = t.cpu()
            T.append(Ti)
            kf.update(center)
            frame_id = frame_id + temporal_dir

        if temporal_dir == -1:
            return trace[::-1], centers[::-1], T[::-1], errors[::-1], selected_indices[::-1]
        else:
            return trace, centers, T, errors, selected_indices

    def track(self, _points, _velocity):
        velocity = _velocity.clone().cuda()

        # forward
        frame_id = _points[0, -1].long().item()
        trace_f, centers_f, T_f, errors_f, selected_indices_f = \
            self.track_dir(_points, velocity, frame_id, self.num_frames-1, 1)
        
        # backward
        frame_id = _points[0, -1].long().item()
        trace_b, centers_b, T_b, errors_b, selected_indices_b = \
            self.track_dir(_points, velocity, frame_id, 0, -1)
        
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
