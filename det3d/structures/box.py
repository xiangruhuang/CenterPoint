import pickle
import numpy as np
from det3d.core.bbox import box_np_ops
import torch

class Box:
    def __init__(self, frame, box, corners):
        self.frame = frame
        self.box = box
        self.corners = corners
        
