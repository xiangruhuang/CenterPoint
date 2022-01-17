import numpy as np
import torch

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

@PIPELINES.register_module
class TraceRandomRotation(object):
    def __init__(self, cfg=None, **kwargs):
        pass

    def __call__(self, res, info):
        theta = np.random.uniform()*2*np.pi
        theta = torch.tensor(theta, dtype=torch.float64)
        points = res['lidar']['points'].double()
        R = torch.eye(2, dtype=torch.float64)
        cost, sint = torch.cos(theta), torch.sin(theta)
        R[:2, :2] = torch.tensor([[cost, -sint],
                                  [sint,  cost]], dtype=torch.float64)
        
        points[:, :2] = points[:, :2] @ R.T
        res['lidar']['points'] = points.float()

        return res, info
