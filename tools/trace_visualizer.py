from det3d.core.utils.visualization import Visualizer
import torch, numpy as np
import sys
from det3d.core.bbox import box_np_ops

def visualize(path):
    vis = Visualizer([], [])
    vis.clear()
    trace_id = int(path.split('/')[-1].split('.')[0].split('_')[-1])
    data=torch.load(path)

    corners = []
    classes = []
    points = []

    for fid in data['T'].keys():
        T_f = data['T'][fid]
        box = data['box'][fid]
        cls = data['cls'][fid]
        box = box[np.newaxis, :]
        corner = box_np_ops.center_to_corner_box3d(box[:, :3], box[:, 3:6],
                                                   -box[:, -1], axis=2)[0]
        R_f, t_f = T_f[:3, :3], T_f[:3, 3]
        corner = corner @ R_f.T + t_f
        corners.append(torch.tensor(corner))
        classes.append(cls)
        points.append(data['points'][fid].cpu())

    classes = torch.tensor(classes)
    corners = torch.stack(corners, dim=0) 
    vis.boxes(f'box-{trace_id}', corners, classes)
    for i in range(corners.shape[0]):
        center = corners[i].mean(0)
        p_center = points[i].mean(0)
        t = p_center[:3] - center
        corners[i] += t
    points = torch.cat(points, dim=0)
    vis.boxes(f'box-pred-{trace_id}', corners, classes, enabled=False)
    ps_p = vis.pointcloud(f'points-{trace_id}', points[:, :3], radius=2e-3, color=(1,1,0))
    ps_p.add_scalar_quantity('frame % 2', points[:, -1] % 2)
    vis.show()

for i, path in enumerate(sys.argv[1:]):
    print(i)
    visualize(path)
