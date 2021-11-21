from ..registry import PIPELINES
from det3d.structures import Sequence
import os
import torch

@PIPELINES.register_module
class LoadLiDARSequence(object):
    def __init__(self, debug=False, load_temp=False):
        self.debug = debug
        self.load_temp = load_temp

    def __call__(self, res, info):
        import time
        start_time = time.time()
        seq_id = int(info['path'].split('/')[-1].split('_')[1])
        save_path = f'/mnt/xrhuang/centerpoint/CenterPoint/{seq_id}.pt'
        if os.path.exists(save_path) and self.load_temp:
            seq = torch.load(save_path)
        else:
            seq = Sequence(info)
            seq.toglobal()
            if self.load_temp:
                torch.save(seq, save_path)

        res['lidar_sequence'] = seq
        end_time = time.time()
        if self.debug:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            p = seq.points4d()
            ps_p = vis.pointcloud('points', p[:, :3])
            vis.pc_scalar('points', 'frame % 2', p[:, -1] % 2)
            vis.boxes('box-original', seq.corners(), seq.classes())
            print(f'load lidar sequence: time={end_time-start_time:.4f}')
            #vis.save('/afs/csail.mit.edu/u/x/xrhuang/public_html/load_lidar_sequence.pth')
            vis.show()

        return res, info
