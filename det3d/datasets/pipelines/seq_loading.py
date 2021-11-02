from ..registry import PIPELINES
from det3d.structures import Sequence

@PIPELINES.register_module
class LoadLiDARSequence(object):
    def __init__(self, visualize=False):
        self.visualize = visualize

    def __call__(self, res, info):
        seq = Sequence(info)
        seq.toglobal()
        res['lidar_sequence'] = seq
        if self.visualize:
            from det3d.core.utils.visualization import Visualizer
            vis = Visualizer([], [])
            p = seq.points4d()
            ps_p = vis.pointcloud('points', p[:, :3])
            ps_p.add_scalar_quantity('frame % 2', p[:, -1] % 2)
            vis.boxes('box-original', seq.corners(), seq.classes())
            vis.show()

        return res, info
