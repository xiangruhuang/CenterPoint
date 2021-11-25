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
        self.debug=debug
        self.ht = HashTable(2000000)
        self.vis = Visualizer([], [])

    def register(self, points, points_all, temporal_offset):
        voxel_size = self.corres_voxel_size.cuda()
        frame_id = points[0, -1].long().item() + temporal_offset
        frame_indices = torch.where(points_all[:, -1] == frame_id)[0]
        frame = points_all[frame_indices]
        ep, ef = self.ht.find_corres(frame, points, voxel_size, 
                                     temporal_offset).long()
        if ep.shape[0] * 2 < points.shape[0]:
            t = torch.zeros(3).float().cuda()
            error = 1e10
        else:
            t = (frame[ef, :3] - points[ep, :3]).mean(0)
            error = (frame[ef, :3]-points[ep, :3]-t).norm(p=2, dim=-1).mean()
        
        return torch.eye(3).float().cuda(), t, error

    def check_status(self, centers, center, error):
        if error > self.threshold:
            return False

        if len(centers) > 1:
            v = center - centers[-1]
            v_last = centers[-1] - centers[-2]
            acc = (v_last - v).norm(p=2, dim=-1)
            #dir_last = v_last / (v_last.norm(p=2, dim=-1) + 1e-6)
            #dir_cur = v / (v.norm(p=2, dim=-1) + 1e-6)
            #angle = (dir_last * dir_cur).sum().arccos() / np.pi * 180
            if acc > self.acc_threshold:
                #print(f'acc {acc}')
                return False
            #if angle > self.angle_threshold:
            #    print(f'angle {angle}')
            #    break

        return True

    def track_dir(self, _points, points_all, velocity,
                  frame_id, temporal_dir = 1):
        """
        
        Returns:
            trace, centers, T, errors
        """
        num_frames = points_all[:, -1].max().long()+1
        if temporal_dir == -1:
            end_frame = 0
        elif temporal_dir == 1:
            end_frame = num_frames-1
        else:
            raise ValueError('Temporal Dir needs to be in {-1, 1}')
        if frame_id + temporal_dir*2 > end_frame:
            return [], [], [], []
        points = _points.clone().cuda()
        center = torch.zeros(6)
        center[:3] = points.mean(0)[:3]
        center[3:] = velocity.mean(0)
        kf = KalmanFilter(center, **self.kf_config)

        trace, centers, T, errors = [], [], [], []
        while frame_id + temporal_dir <= end_frame:
            center_next = kf.predict()
            t0 = (center_next - center).cuda()
            points[:, :3] = points[:, :3] + t0
            R, t, error = self.register(points, points_all, 1)
            t = t + t0
            points[:, :3] = points[:, :3] @ R.T + t
            center = points.mean(0)[:3].cpu()
            if not self.check_status(centers, center, error):
                break
            trace.append(points.cpu())
            errors.append(error)
            centers.append(center)
            Ti = torch.eye(4)
            Ti[:3, :3] = R
            Ti[:3, 3] = t
            T.append(Ti)
            kf.update(center)
            frame_id = frame_id + 1
            points[:, -1] += 1

        if temporal_dir == -1:
            return trace[::-1], centers[::-1], T[::-1], errors[::-1]
        else:
            return trace, centers, T, errors

    def track(self, _points, points_all, _velocity):
        velocity = _velocity.clone().cuda()
        points_all = points_all.cuda()
        num_frames = points_all[:, -1].max().long()+1
        frame_id = _points[0, -1].long().item()        

        # forward
        trace_f, centers_f, T_f, errors_f = \
            self.track_dir(_points, points_all, velocity, frame_id, 1)
        
        # backward
        trace_b, centers_b, T_b, errors_b = \
            self.track_dir(_points, points_all, velocity, frame_id, -1)
        
        # merge
        trace = trace_b + [_points.cpu()] + trace_f
        centers = centers_b + [_points.mean(0)[:3].cpu()] + centers_f
        T = T_b + [torch.eye(4)] + T_f
        errors = errors_b + [torch.tensor(0.0)] + errors_f

        errors = torch.tensor(errors).cuda()
        centers = torch.stack(centers, dim=0).cuda()
        return trace, centers, T, errors

    def fake_track(self, cluster, all_points, velocity, corres0):
        cluster = cluster.cuda()
        all_points = all_points.cuda()
        velocity = velocity.cuda()
        corres0 = torch.tensor(corres0).cuda()
        voxel_size = torch.tensor([1., 1., 1., 1]).cuda()
        trace, centers, errors = [cluster.cpu()], [cluster.mean(0)[:3]], []
        num_frames = all_points[:, -1].max().long().item() + 1

        # forward
        corres = corres0.clone()
        deformed = cluster.clone()
        frame_id = cluster[0, -1].long().item()
        while frame_id + 1 < num_frames:
            deformed[:, :3] += velocity[corres]
            ea, ed = self.ht.find_corres(deformed, all_points, voxel_size, 1).long()
            #ed, ea = query_corres(deformed, all_points, voxel_size, 1).T.long()
            if ed.shape[0] == 0:
                break
            err = (deformed[ed] - all_points[ea])[:, :3].norm(p=2, dim=-1).mean()
            if err > self.reg_threshold:
                print(f'reg {err}')
                break
            center = deformed[ed, :3].mean(0)
            if len(centers) > 1:
                v = centers[-1] - center
                v_last = centers[-2] - centers[-1]
                acc = (v_last - v).norm(p=2, dim=-1)
                dir_last = v_last / (v_last.norm(p=2, dim=-1) + 1e-6)
                dir_cur = v / (v.norm(p=2, dim=-1) + 1e-6)
                angle = (dir_last * dir_cur).sum().arccos() / np.pi * 180
                if acc > self.acc_threshold:
                    print(f'acc {acc}')
                    break
                if angle > self.angle_threshold:
                    print(f'angle {angle}')
                    break
            centers.append(center)
            errors.append(err)
            deformed = all_points[ea].clone()
            corres = ea
            trace.append(deformed.cpu())
            frame_id += 1
        trace = trace[::-1]

        # backward 
        corres = corres0.clone()
        deformed = cluster.clone()
        frame_id = cluster[0, -1].long()
        while frame_id - 1 >= 0:
            deformed[:, :3] -= velocity[corres]
            ea, ed = self.ht.find_corres(deformed, all_points, voxel_size, -1).long()
            #ed, ea = query_corres(deformed, all_points, voxel_size, -1).T.long()
            if ed.shape[0] == 0:
                break
            err = (deformed[ed] - all_points[ea])[:, :3].norm(p=2, dim=-1).mean()
            if err > self.reg_threshold:
                print(f'reg {err}')
                break
            center = deformed[ed, :3].mean(0)
            if len(centers) > 1:
                v = centers[-1] - center
                v_last = centers[-2] - centers[-1]
                acc = (v_last - v).norm(p=2, dim=-1)
                dir_last = v_last / (v_last.norm(p=2, dim=-1) + 1e-6)
                dir_cur = v / (v.norm(p=2, dim=-1) + 1e-6)
                angle = (dir_last * dir_cur).sum().arccos() / np.pi * 180
                if acc > self.acc_threshold:
                    print(f'acc {acc}')
                    break
                if angle > self.angle_threshold:
                    print(f'angle {angle}')
                    break
            centers.append(center)
            errors.append(err)
            deformed = all_points[ea].clone()
            corres = ea
            trace.append(deformed.cpu())
            frame_id -= 1

        if len(errors) > 0:
            errors = torch.tensor(errors)
        else:
            errors = torch.zeros(0)
        centers = torch.stack([t.mean(0)[:3] for t in trace], dim=0).cpu()

        return trace, centers, errors

    def find_trace(self, voxels, voxels_velo, num_graphs, graph_idx, seq):

        def check_trace_status(self, trace, centers):
            if len(trace) < 10:
                return False
            lengths = (centers[1:, :3] - centers[:-1, :3]).norm(p=2, dim=-1)
            travel_dist = lengths.sum()
            mean_velo = (centers[0, :3] - centers[-1, :3]
                         ).norm(p=2, dim=-1) / lengths.shape[0]
            if travel_dist < self.min_travel_dist:
                return False
            if mean_velo < self.min_mean_velocity:
                return False
            return True

        def update(self, voxels, trace, voxel_weight, voxel_dr):
            et, ev = voxel_graph(voxels, trace[:, :4], self.voxel_size,
                                 0, 16).T.long()
            dist = (trace[et, :3] - voxels[ev, :3]).norm(p=2, dim=-1)
            weight = np.exp(-(dist / 0.3)**2/2.0)
            weight_dr = weight[:, np.newaxis] * trace[:, 4:]
            voxel_weight += scatter(weight, ev, dim=0,
                                    dim_size=voxels.shape[0], reduce='sum')
            voxel_dr += scatter(weight_dr, ev, dim=0,
                                dim_size=voxels.shape[0], reduce='sum')

        vis = Visualizer([], [])
        ps_v = vis.pointcloud('voxels', voxels[:, :3])
        vnorm = voxels_velo.norm(p=2, dim=-1)
        ps_v.add_scalar_quantity('velocity', vnorm)
        voxel_weight = torch.zeros(voxels.shape[0])
        voxel_dr = torch.zeros(voxels.shape[0], 3)
        vis.boxes('box', seq.corners(), seq.classes())
        
        for i in range(933, num_graphs):
            if (i + 1) % 10000 == 0:
                import ipdb; ipdb.set_trace()
                vis.show()
            mask = (graph_idx == i)
            indices = np.where(mask)[0]
            voxels_i = voxels[indices].clone()
            avg_velo = voxels_velo[indices].mean(0).norm(p=2)
            frame_id = voxels_i[0, -1].long()
            if (avg_velo > 0.05) and indices.shape[0] > 5 and (frame_id + 5 < len(seq.frames)):
                trace, centers, T, errors = \
                        self.track(voxels_i, voxels, voxels_velo)
                if not self.check_trace_status(trace, centers):
                    continue
                trace = torch.cat(trace, dim=0)
                print(f'pass {i:05d}')
                self.update(voxels, trace, voxel_weight, voxel_dr)
                #ps_trace = vis.pointcloud(f'trace-{i}', trace[:, :3], radius=6e-4)
                #ps_trace.add_scalar_quantity('frame', trace[:, -1])
                #ps_cluster = vis.pointcloud(f'cluster-{i}', voxels_i[:, :3], radius=4e-4)
                #ps_c = vis.trace(f'center-trace-{i}', centers[:, :3], enabled=False, radius=4e-4)
                #ps_c.add_scalar_quantity('error', errors.cpu(), defined_on='edges')
                #ps_c.add_scalar_quantity('length', lengths.cpu(), defined_on='edges')
                ps_v.add_scalar_quantity('voxel weight', voxel_weight)
                ps_v.add_scalar_quantity('voxel dr', voxel_dr / voxel_weight.unsqueeze(-1))
                vis.show()
        vis.show()

    def __call__(self, res, info):
        #seq = res['lidar_sequence']
        #import ipdb; ipdb.set_trace()
        res = torch.load('obj_tracking.pt')
        seq = res['lidar_sequence']
        voxels, voxels_velo = res['voxels'], res['voxels_velo']
        ev, ep = res['vp_edges']
        num_graphs, graph_idx = res['num_graphs'], res['graph_idx']
        import ipdb; ipdb.set_trace()
        self.find_trace(voxels, voxels_velo, num_graphs, graph_idx, seq)
        
        return res, info 
