from det3d.core.utils.visualization import Visualizer
import torch, numpy as np
import sys
from det3d.core.bbox import box_np_ops

vis = Visualizer([], [])
data=torch.load(sys.argv[1])

corners = []
classes = []
points = []

for fid in data['T'].keys():
    T_f = data['T'][fid]
    box = data['box'][fid]
    cls = data['cls'][fid]
    box = box[np.newaxis, :]
    corner = box_np_ops.center_to_corner_box3d(box[:, :3], box[:, 3:6],
                                               box[:, -1], axis=2)[0]
    R_f, t_f = T_f[:3, :3], T_f[:3, 3]
    corner = corner @ R_f.T + t_f
    corners.append(torch.tensor(corner))
    classes.append(cls)
    points.append(data['points'][fid])

classes = torch.tensor(classes)
corners = torch.stack(corners, dim=0) 
points = torch.cat(points, dim=0)
vis.boxes('box', corners, classes)
vis.pointcloud('points', points[:, :3], radius=2e-3)
vis.show()
