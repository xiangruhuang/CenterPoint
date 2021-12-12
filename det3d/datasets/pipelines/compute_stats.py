from ..registry import PIPELINES
import numpy as np
import os
from torch_scatter import scatter
from torch_cluster import radius_graph
import torch
import scipy
from scipy.sparse import csr_matrix
from det3d.ops.primitives.primitives_cpu import (
    voxelization, 
)
from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import (
    points_in_convex_polygon_3d_jit,
)

def solve(p, q):
    p = torch.tensor(p).clone()
    q = torch.tensor(q).clone()
    q = q.double()
    R = torch.eye(3).double()
    shift = q.mean(0) - p.mean(0)
    p = p - p.mean(0)
    q = q - q.mean(0)
    for itr in range(3):
        t = (q - p @ R.T).mean(0)
        M = ( p[:, :2].unsqueeze(-1) @ (q-t)[:, :2].unsqueeze(-2) ).sum(0)
        U, S, V = M.double().svd()
        R2 = V @ U.T
        if R2.det() < 0:
            R2 = V.clone()
            R2[:, -1] *= -1
            R2 = R2 @ U.T
        R[:2, :2] = R2
        error = (q - p @ R.T - t).norm(p=2, dim=-1).mean()
    t = t + shift
    return R, t, error

@PIPELINES.register_module
class ComputeStats(object):
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.num_boxes = [0, 0, 0]
        self.num_unique_objects = [0, 0, 0]
        self.num_unique_moving_objects = [0, 0, 0]
        self.num_moving_boxes = [0, 0, 0]
        self.num_points_in_box = [[], [], []]
        self.num_points_in_moving_box = [[], [], []]
        self.velo = [[], [], []]
        self.turning_angle = [[], [], []]
        self.num_isolated = [[], [], []]

    def __call__(self, res, info):
        seq = res['lidar_sequence']
        self.seq_id = seq.seq_id
        trace_dict = {}
        self.initialize()
        for f in seq.frames:
            surfaces = box_np_ops.corner_to_surfaces_3d(f.corners)
            indices = points_in_convex_polygon_3d_jit(f.points[:, :3], surfaces)
            num_points_in_boxes = indices.astype(np.int32).sum(0)
            for box_id, token in enumerate(f.tokens):
                cls = f.classes[box_id]
                if trace_dict.get(token, None) is None:
                    trace_dict[token] = []
                box_dict = dict(frame_id = f.frame_id,
                                corners = f.corners[box_id],
                                cls = f.classes[box_id],
                                num_points = num_points_in_boxes[box_id])
                trace_dict[token].append(box_dict)

        offsets = []
        offset = 0
        for f in seq.frames:
            offsets.append(offset)
            offset += f.points.shape[0]
        offsets.append(offset)
        num_graphs, graph_idx = res['num_graphs'], res['graph_idx']
        ev, ep = res['vp_edges']
        graph_idx_by_point = scatter(torch.tensor(graph_idx[ev]).long(), ep,
                                     dim=0, reduce='max',
                                     dim_size=offsets[-1])
        graph_idx_per_frame = [graph_idx_by_point[offsets[frame_id]:offsets[frame_id+1]]
                                 for frame_id, f in enumerate(seq.frames)]

        for token in trace_dict.keys():
            box_trace = trace_dict[token]
            abs_travel_dist = 0
            cls = box_trace[0]['cls']
            self.num_unique_objects[cls] += 1
            self.num_boxes[cls] += len(box_trace)

            # check if moving
            for i in range(1, len(box_trace)):
                last_box = box_trace[i-1]
                box = box_trace[i]
                abs_travel_dist += np.linalg.norm(
                                       last_box['corners'] - box['corners'],
                                       ord=2, axis=-1).sum()
            for box_dict in box_trace:
                self.num_points_in_box[cls].append(box_dict['num_points'])
                
            velos = []
            thetas = []
            for box_id, box_dict in enumerate(box_trace):
                velo = []
                theta = []
                if box_id > 0:
                    p = box_trace[box_id-1]['corners']
                    q = box_dict['corners']
                    velo.append(np.linalg.norm(p - q, ord=2, axis=-1).mean())
                    R, t, error = solve(p, q)
                    theta.append(R[0, 0].clip(-1, 1).arccos())
                    assert (not np.isnan(theta[-1]))
                if box_id < len(box_trace) - 1:
                    p = box_dict['corners']
                    q = box_trace[box_id+1]['corners']
                    velo.append(np.linalg.norm(p - q, ord=2, axis=-1).mean())
                    R, t, error = solve(p, q)
                    theta.append(R[0, 0].clip(-1, 1).arccos())
                    assert (not np.isnan(theta[-1]))
                velos.append(np.mean(velo))
                thetas.append(np.mean(theta))

            if (abs_travel_dist > 0.5) and (np.mean(velos) > 0.05):
                self.num_moving_boxes[cls] += len(box_trace)
                self.num_unique_moving_objects[cls] += 1

                for box_dict in box_trace:
                    self.num_points_in_moving_box[cls].append(box_dict['num_points'])

                self.velo[cls] += velos
                self.turning_angle[cls] += thetas

                # check if any isolated cluster
                isolated = 0
                for box_dict in box_trace:
                    frame_id = box_dict['frame_id']
                    graph_idx_this = graph_idx_per_frame[frame_id]
                    f = seq.frames[frame_id]
                    surfaces = box_np_ops.corner_to_surfaces_3d(
                                   box_dict['corners'][np.newaxis, ...])
                    indices = points_in_convex_polygon_3d_jit(f.points[:, :3],
                                                              surfaces)[:, 0]
                    inbox_indices = np.where(indices)[0]
                    if inbox_indices.shape[0] == 0:
                        continue
                    graph_id = graph_idx_this[inbox_indices[0]]
                    flag = (graph_idx_this[inbox_indices] == graph_id).all()
                    flag2 = np.where(graph_idx_this == graph_id)[0].shape[0] == inbox_indices.shape[0]
                    
                    if flag and flag2:
                        isolated += 1

                self.num_isolated[cls].append((isolated, len(box_trace)))

        save_dict = dict(
            num_boxes = self.num_boxes,
            num_unique_objects = self.num_unique_objects,
            num_unique_moving_objects = self.num_unique_moving_objects,
            num_moving_boxes = self.num_moving_boxes,
            num_points_in_box = self.num_points_in_box,
            num_points_in_moving_box = self.num_points_in_moving_box,
            velo = self.velo,
            turning_angle = self.turning_angle,
            num_isolated = self.num_isolated,
        )
        torch.save(save_dict,
                   f'work_dirs/stats/seq_{seq.seq_id}.pt')

        return res, info
