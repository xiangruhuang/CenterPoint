import os.path as osp
import warnings
import numpy as np
from functools import reduce
import torch

import pycocotools.mask as maskUtils
from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES
from torch_scatter import scatter
from det3d.core.bbox import box_np_ops

@PIPELINES.register_module
class Visualization(object):
    def __init__(self, points=True, boxes=True):
        self.vis_points = points
        self.vis_boxes = boxes

    def __call__(self, res, info):
        from det3d.core.utils.visualization import Visualizer
        vis = Visualizer([0.2, 0.2], [-75.2, -75.2, 75.2, 75.2])
        if self.vis_points:
            points = res['lidar']['points']
            ps_p = vis.pointcloud('points', points[:, :3])
        if self.vis_boxes:
            gt_boxes = info['gt_boxes']
            corners = box_np_ops.center_to_corner_box3d(
                gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1],
                axis=2
            )
            labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
            vis.boxes('box', corners, labels)
        vis.show()
        vis.clear()
        
        return res, info
