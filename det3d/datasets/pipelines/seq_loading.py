from ..registry import PIPELINES
from det3d.structures import Sequence
import os
import torch

@PIPELINES.register_module
class LoadLiDARSequence(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq = Sequence(info)
        seq.toglobal()
        res['lidar_sequence'] = seq
        end_time = time.time()
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            p = seq.points4d()
            ps_p = vis.pointcloud('points', p[:, :3])
            ps_p.add_scalar_quantity('frame % 2', p[:, -1] % 2)
            vis.boxes('box-original', seq.corners(), seq.classes())
            print(f'load lidar sequence: time={end_time-start_time:.4f}')
            vis.show()

        return res, info
