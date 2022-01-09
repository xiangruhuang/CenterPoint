import numpy as np
import torch
import glob

import pickle
#from det3d.core.utils.visualization import Visualizer
from torch_scatter import scatter
from det3d.ops.iou3d_nms import iou3d_nms_utils 
from det3d.core.bbox import box_np_ops
import math

trace_files = glob.glob('data/Waymo/train/traces2/*.pt')
with open('work_dirs/waymo_trace_classifer_training/prediction.pkl', 'rb') as fin:
    prediction = pickle.load(fin)

#vis = Visualizer([], [])
cls = 1
box_size = {0: torch.tensor([4.8557, 2.1343, 1.7822]),
            1: torch.tensor([0.9858, 0.8887, 1.7669]),
            2: torch.tensor([1.8623, 0.8435, 1.7970])}
box_std = {0: torch.tensor([0.9531, 0.2415, 0.3746]),
           1: torch.tensor([0.1755, 0.1372, 0.1504]),
           2: torch.tensor([1.8623, 0.8435, 1.7970])
           }
if cls == 1:
    weight_smooth = 0.05
    origin0 = torch.tensor([0, 0, 2.5], dtype=torch.float64)
elif cls == 2:
    weight_smooth = 0.1
    origin0 = torch.tensor([0, 0, 2.5], dtype=torch.float64)
elif cls == 0:
    weight_smooth = 0.1
    origin0 = torch.tensor([0, 0, -0.5], dtype=torch.float64)

box_sizes = []
def away_from_origin(points, corners, origins, classes):
    """Move box corners so that points are in the box and box is away from origin.
    
    Args:
        points (torch.tensor, [N, 4]): 
        corners (torch.tensor, M, 8, 3): 
        origins (M, 3): 
        classes: dummy
    
    Returns:
        shifts (M, 3)
    """
    #vis.pointcloud('points', points[:, :3], radius=2e-3)
    points = points.cuda()
    corners = corners.cuda()
    origins = origins.cuda()
    num_frames = corners.shape[0]
    edges = torch.tensor([[0, 1], [0, 3], [0, 4], [1, 0], [3, 0], [4, 0]]).long()
    lrelu = torch.nn.LeakyReLU(0.0001, inplace=True)
    shifts = torch.nn.Parameter(torch.zeros(num_frames, 3, dtype=torch.float64).cuda())
    frame_ids = points[:, -1].long()
    min_frame_id = frame_ids.min().item()
    max_frame_id = frame_ids.max().item()
    frame_ids = frame_ids - min_frame_id
    
    optimizer = torch.optim.Adam([shifts], lr=1e-3)
    # each plane's normal [N, 6, 3]
    normals = corners[frame_ids][:, edges[:, 1]] - corners[frame_ids][:, edges[:, 0]]
    normals = normals / normals.norm(p=2, dim=-1).unsqueeze(-1)
    
    from_origins = corners.mean(1) - origins
    from_origins = from_origins / (from_origins.norm(p=2, dim=-1).unsqueeze(-1)+1e-6)
    weight = torch.tensor([10, 10, 30.0]).cuda()

    for itr in range(8000):
        optimizer.zero_grad()
        # shift box by shifts [M, 8, 3]
        shifted_corners = corners + shifts.unsqueeze(1)

        # point to plane signed distance [N, 6]
        shift = -((points[:, :3].unsqueeze(1) - shifted_corners[frame_ids][:, edges[:, 0]])*normals).sum(-1)
        loss_in_box = lrelu(shift).sum()
        in_box_ratio = (shift < 0).float().mean()
        
        # loss of away from origin
        if cls == 0:
            loss_away = -((shifted_corners.mean(1) - origins)*from_origins).sum()
        else:
            loss_away = -((shifted_corners.mean(1) - origins)*from_origins)[:, 2].sum()

        # loss smoothness
        shifted_centers = shifted_corners.mean(1)
        loss_smooth = ((shifted_centers[1:] - shifted_centers[:-1])*weight).square().sum()

        # boundary condition
        if cls != 0:
            loss_boundary = shifts[0, :2].square().sum() + shifts[-1, :2].square().sum()
        else:
            loss_boundary = (shifts[0, :2] - shifts[-1, :2]).square().sum()

        loss = loss_away + loss_in_box + weight_smooth*loss_smooth + num_frames*loss_boundary
        loss.backward()
        grad = shifts.grad.norm(p=2, dim=-1).mean()
        shift_norm = shifts.norm(p=2, dim=-1).mean()
        optimizer.step()
        if itr % 1000 == 0:
            print(f'iter={itr}, loss={loss.item():.4f}',
                  f'in_box={loss_in_box.item():.4f}',
                  f'away={loss_away.item():.4f}',
                  f'smooth={loss_smooth.item():.4f}',
                  f'boundary={loss_smooth.item():.4f}',
                  f'in_box_ratio={in_box_ratio:.4f}',
                  f'grad_norm={grad:.4f}, shift={shift_norm:.4f}',
                  )
        

    #vis.boxes('box', shifted_corners.detach().cpu(), classes.cpu())

    return shifts.detach().cpu()
    
iou_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
num_total = 0
num_accurate = np.array([0 for th in iou_thresholds])
for trace_file in trace_files:
    token = trace_file.split('/')[-1].split('.')[0]
    trace = torch.load(trace_file)
    pred = prediction[token]
    gt_cls = trace['gt_cls']
    if gt_cls == cls:
        #vis.clear()
        seq_id = int(trace_file.split('/')[-1].split('.')[0].split('_')[1])
            
        points = trace['points']
        frame_ids = points[:, -1].long()
        min_frame_id = frame_ids.min().item()
        max_frame_id = frame_ids.max().item()
        num_frames = max_frame_id - min_frame_id + 1
        frame_ids -= min_frame_id
        gt_boxes = trace['boxes'].numpy()
        gt_boxes = gt_boxes[:, [0,1,2,3,4,5,-1]]
        
        centers = scatter(points[:, :3], frame_ids, reduce='mean',
                          dim=0, dim_size=num_frames)
        #ps_pc = vis.pointcloud('centers', centers)
        pred_boxes = torch.zeros(num_frames, 7)
        pred_boxes[:, 3:6] = box_size[cls]
        heading_dir = (centers[5:] - centers[:-5]).double()
        heading_dir = scatter(torch.cat([heading_dir, heading_dir], dim=0),
                              torch.cat(
                                  [torch.arange(num_frames-5),
                                   torch.arange(num_frames-5)+5],
                                  dim=0),
                              reduce='mean', dim=0, dim_size=num_frames,
                             )
        heading_dir = heading_dir / heading_dir.norm(p=2, dim=-1).unsqueeze(-1)
        origins, from_origins = [], []
        for frame_id in range(min_frame_id, max_frame_id+1):
            center = centers[frame_id - min_frame_id]
            hdir = heading_dir[frame_id-min_frame_id]
            with open(f'data/Waymo/train/annos/seq_{seq_id}_frame_{frame_id}.pkl', 'rb') as fin:
                annos = pickle.load(fin)

            T = annos['veh_to_global'].reshape(4, 4)
            T = torch.tensor(T, dtype=torch.float64)
            origin = origin0.clone()
            origin = origin @ T[:3, :3].T + T[:3, 3]
            origins.append(origin)
            from_origins.append(center - origin)
            centers[frame_id-min_frame_id] = T[:3, :3].T @ (center - T[:3, 3])
            
            heading_dir[frame_id-min_frame_id] = T[:3, :3].T @ hdir

        from_origins = torch.stack(from_origins, dim=0)
        
        #ps_pc.add_vector_quantity('from origin', from_origins, enabled=True)
        
        origins = torch.stack(origins, dim=0)
        hx, hy = heading_dir[:, :2].T
        heading_angle = np.arctan2(hy, hx)
        
        pred_boxes[:, :3] = centers
        pred_boxes[:, 6] = heading_angle
        pred_boxes = pred_boxes.numpy()
        pred_corners = box_np_ops.center_to_corner_box3d(pred_boxes[:, :3],
                                                         pred_boxes[:, 3:6],
                                                         -pred_boxes[:, -1],
                                                         axis=2)
        pred_corners = torch.tensor(pred_corners).double()
        for frame_id in range(min_frame_id, max_frame_id+1):
            corner = pred_corners[frame_id - min_frame_id]
            with open(f'data/Waymo/train/annos/seq_{seq_id}_frame_{frame_id}.pkl', 'rb') as fin:
                annos = pickle.load(fin)

            T = annos['veh_to_global'].reshape(4, 4)
            T = torch.tensor(T, dtype=torch.float64)
            pred_corner = corner @ T[:3, :3].T + T[:3, 3]
            pred_corners[frame_id-min_frame_id] = pred_corner
            origin = origins[frame_id - min_frame_id]
        gt_corners = trace['corners']
        #vis.boxes('gt', gt_corners, trace['classes'])
        pred_boxes = torch.tensor(pred_boxes)
        #vis.boxes('pred', pred_corners, trace['classes'], enabled=False)
        # shift away from origin
        shifts = away_from_origin(points, pred_corners, origins, trace['classes'])
        pred_corners += shifts.unsqueeze(1)
        
        for frame_id in range(min_frame_id, max_frame_id+1):
            shift = shifts[frame_id - min_frame_id]
            with open(f'data/Waymo/train/annos/seq_{seq_id}_frame_{frame_id}.pkl', 'rb') as fin:
                annos = pickle.load(fin)

            T = annos['veh_to_global'].reshape(4, 4)
            T = torch.tensor(T, dtype=torch.float64)
            shift = shift @ T[:3, :3]
            shifts[frame_id - min_frame_id] = shift
            pred_boxes[frame_id - min_frame_id, :3] += shift

        ious = torch.diag(iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes.cuda().float(), torch.tensor(gt_boxes).cuda().float()))
        num_total += ious.shape[0]
        for ii, iou_threshold in enumerate(iou_thresholds):
            num_accurate[ii] += (ious > iou_threshold).float().sum()
        print(f'iou acc={num_accurate/num_total}')

        #vis.boxes('shifted-pred', pred_corners, trace['classes'], enabled=True)
        #vis.pointcloud('points', trace['points'][:, :3], radius=2e-3)
