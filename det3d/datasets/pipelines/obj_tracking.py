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
        self.tracker = KalmanTracker(self.ht, kf_config,
                                     threshold, acc_threshold,
                                     voxel_size,
                                     corres_voxel_size, debug)
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

    def motion_sync(self, voxels, voxels_velo, num_graphs, graph_idx, vp_edges, seq):
        points = torch.tensor(seq.points4d(), dtype=torch.float32)
        ev, ep = vp_edges
        if self.debug:
            vis = Visualizer([], [])
            ps_p = vis.pointcloud('points', points[:, :3])
            vis.boxes('box', seq.corners(), seq.classes())
        if self.velocity:
            points_velo = torch.zeros(points.shape[0], 3)
            points_velo[ep] = voxels_velo[ev]
        else:
            points_velo = torch.zeros(points.shape[0], 3)
        self.tracker.set_points(points)
        graph_idx_by_point = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                     dim=0, reduce='max',
                                     dim_size=points.shape[0])
        trace_count = 0

        for i in range(num_graphs):
            indices = torch.where(graph_idx_by_point == i)[0]
            points_i = points[indices].clone()
            avg_velo = points_velo[indices].mean(0).norm(p=2)
            frame_id = points_i[0, -1].long()
            if (avg_velo > self.min_velo) and (points_i.shape[0] >= 10) \
                and (points_i.shape[0] < 500):
                if self.debug:
                    print(f'cluster {i}: ', end="")
                    ps_cluster = vis.pointcloud(f'cluster-{i}', points_i[:, :3], radius=3e-4,
                                                enabled=False)
                t0 = time.time()
                trace, centers, T, errors, selected_indices = \
                        self.tracker.track(points_i, points_velo[indices])
                if len(trace) > 0:
                    print(f'average t={(time.time()-t0)/len(trace):.4f}')
                if not self.check_trace_status(trace, centers):
                    continue
                trace_i = torch.cat([points_i, points[selected_indices]],
                                     dim=0)
                trace = torch.cat(trace, dim=0)
                avg_time = (time.time()-t0)/(i+1)
                eta = avg_time * (num_graphs - i)
                print(f'pass {i:05d}, time={avg_time:.4f}, ETA={eta:.4f}, '\
                      f'num_points={points.shape[0]}')
                
                if self.debug:
                    ps_trace = vis.pointcloud(f'trace-{i}', trace[:, :3],
                                              radius=3e-4, enabled=False)
                    ps_trace.add_scalar_quantity('frame', trace[:, -1])
                    ps_c = vis.trace(f'center-trace-{i}', centers[:, :3],
                                     enabled=False, radius=3e-4)
                    ps_c.add_scalar_quantity('error', errors.cpu(), defined_on='nodes')
                    vis.pointcloud(f'selected-trace-{i}', trace_i[:, :3], radius=3e-4)
                    ps_p = vis.pointcloud('points', points[:, :3])

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
