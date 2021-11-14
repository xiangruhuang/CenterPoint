from ..registry import PIPELINES
from collections import defaultdict
from det3d.structures import Sequence
import os
import torch
import numpy as np
import pickle
from det3d.core.bbox import box_np_ops

@PIPELINES.register_module
class FindMovingBoxes(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = res['lidar_sequence']
        object_traces = seq.object_traces()
        frame_dicts = [{'frame_id': i,
                        'veh_to_global': f.T.reshape(-1),
                        'objects': [],
                        'frame_name': f.frame_name,
                        'scene_name': f.scene_name,
                        } for i, f in enumerate(seq.frames)]
        for trace_dict in object_traces:
            if trace_dict['frame_ids'].shape[0] < 20:
                continue
            corners = trace_dict['corners']
            dist = torch.tensor(corners[0] - corners[-1]).norm(p=2, dim=-1).mean()
            if dist < 5:
                continue
            for i, frame_id in enumerate(trace_dict['frame_ids']):
                obj = dict(id=len(frame_dicts[frame_id]['objects']),
                           name=trace_dict['tokens'][i],
                           label=trace_dict['classes'][i],
                           box=trace_dict['boxes'][i],
                           global_speed=trace_dict['global_speed'][i],
                           global_accel=trace_dict['global_accel'][i],
                           num_points=100,
                          )
                frame_dicts[frame_id]['objects'].append(obj)

        seq_id = seq.seq_id
        for fid, frame_dict in enumerate(frame_dicts):
            filename=f'data/Waymo/train_moving/annos/seq_{seq_id}_frame_{fid}.pkl'
            try:
                with open(filename, 'wb') as fout:
                    pickle.dump(dict(frame_dict), fout)
            except Exception as e:
                print(filename, e)
                import ipdb; ipdb.set_trace()
                print(e)
        print(f'saved sequence {seq_id}')
        end_time = time.time()
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            vis.clear()
            p = seq.points4d()
            ps_p = vis.pointcloud('points', p[:, :3])
            ps_p.add_scalar_quantity('frame % 2', p[:, -1] % 2)
            for fid, frame_dict in enumerate(frame_dicts):
                objects = frame_dict['objects']
                boxes = np.array([o['box'] for o in objects])
                corners = box_np_ops.center_to_corner_box3d(
                    boxes[:, :3], boxes[:, 3:6], -boxes[:, -1], axis=2)
                T = frame_dict['veh_to_global'].reshape(4, 4)
                corners = (corners.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
                           ).reshape(-1, 8, 3)
                classes = np.array([o['label'] for o in objects])
                vis.boxes(f'box-frame-{fid}', corners, classes)
            vis.boxes(f'box-all', seq.corners(), seq.classes())
            print(f'finding moving boxes: time={end_time-start_time:.4f}')
            vis.show()

        return res, info
