from det3d.core.utils.visualization import Visualizer
import torch, numpy as np
import sys
from det3d.core.bbox import box_np_ops
from torch_scatter import scatter

vis = Visualizer([], [])

def visualize(path):
    trace_id = int(path.split('/')[-1].split('.')[0].split('_')[-1])
    data=torch.load(path)

    corners = []
    classes = []
    points = []

    data = sorted(data, key = lambda x: x['points'].shape[0], reverse=True)
    
    import ipdb; ipdb.set_trace()
    for i, trace in enumerate(data):
        points = trace['points']
        corners = trace['corners']
        classes = trace['classes']
        frame_indices = points[:, -1].long()
        num_frames = frame_indices.max().item() + 1
        centers = scatter(points[:, :3], frame_indices,
                          dim_size=num_frames, dim=0, reduce='sum')
        weights = scatter(torch.ones_like(points[:, -1]), frame_indices,
                          dim_size=num_frames, dim=0, reduce='sum')
        mask = weights > 0
        centers[mask] = centers[mask] / weights[mask].unsqueeze(-1)
        print(f'processing {i} / {len(data)}')
        #vis.trace(f'center-trace-{i}', centers[mask])
        vis.pointcloud(f'points-{i}', points[:, :3], enabled=True)
        vis.boxes(f'box-{i}', corners, classes, enabled=True)
        if (i + 1) % 1 == 0:
            import ipdb; ipdb.set_trace()
            vis.show()
            pass
    vis.show()

    for i, fid in enumerate(data['T'].keys()):
        T_f = data['T'][fid]
        box = data['box'][fid]
        cls = data['cls'][fid]
        box = box[np.newaxis, :]
        if (i == 0) and (cls==1):
            print(cls, box[:, 3:6])
        corner = box_np_ops.center_to_corner_box3d(box[:, :3], box[:, 3:6],
                                                   -box[:, -1], axis=2)[0]
        R_f, t_f = T_f[:3, :3], T_f[:3, 3]
        corner = corner @ R_f.T + t_f
        point_center = data['points'][fid].cpu().mean(0)[:3]
        #if torch.tensor(point_center - corner.mean(0)).norm(p=2, dim=-1) > 4.0:
        #    continue
        corners.append(torch.tensor(corner))
        classes.append(cls)
        points.append(data['points'][fid].cpu())

    if len(points) == 0:
        return
    classes = torch.tensor(classes)
    corners = torch.stack(corners, dim=0) 
    vis.boxes(f'box-{trace_id}', corners, classes)
    for i in range(corners.shape[0]):
        center = corners[i].mean(0)
        p_center = points[i].mean(0)
        t = p_center[:3] - center
        corners[i] += t
    points = torch.cat(points, dim=0)
    #vis.boxes(f'box-pred-{trace_id}', corners, classes, enabled=False)
    ps_p = vis.pointcloud(f'points-{trace_id}', points[:, :3], radius=2e-4, color=(1,1,0))
    ps_p.add_scalar_quantity('frame % 2', points[:, -1] % 2)
    import ipdb; ipdb.set_trace()

for i, path in enumerate(sys.argv[1:]):
    visualize(path)
vis.show()
