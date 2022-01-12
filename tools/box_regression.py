import numpy as np
import torch
import glob

import pickle
from det3d.core.utils.visualization import Visualizer
from torch_scatter import scatter
from det3d.ops.iou3d_nms import iou3d_nms_utils 
from det3d.core.bbox import box_np_ops
import math
import argparse

vis = Visualizer([], [])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)

    args = parser.parse_args()

    return args

def away_from_origin(points, corners, origins, classes, cls):
    """Move box corners so that points are in the box and box is away from origin.
    
    Args:
        points (torch.tensor, [N, 4]): 
        corners (torch.tensor, M, 8, 3): 
        origins (M, 3): 
        classes: dummy
    
    Returns:
        shifts (M, 3)
    """
    
    if cls == 1:
        weight_smooth = 0.05
    elif cls == 2:
        weight_smooth = 0.1
    elif cls == 0:
        weight_smooth = 0.1

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
    
    weight = torch.tensor([10, 10, 30.0]).cuda()
    from_origins = (corners.mean(1) - origins)
    from_origins = from_origins / from_origins.norm(p=2, dim=-1).unsqueeze(-1)
    if args.visualize:
        ps_ori = vis.trace('center-trace', corners.mean(1).detach().cpu().numpy())
        ps_ori.add_vector_quantity('from origin', from_origins.detach().cpu().numpy())

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
            loss_away = -((shifted_corners.mean(1) - origins)*from_origins).sum(-1).sum()
        else:
            loss_away = -(shifted_corners.mean(1) - origins).square()[:, 2].sqrt().sum()

        # loss smoothness
        shifted_centers = shifted_corners.mean(1)
        loss_smooth = ((shifted_centers[1:] - shifted_centers[:-1])*weight).square().sum()

        # boundary condition
        if cls != 0:
            loss_boundary = shifts[0, :2].square().sum() + shifts[-1, :2].square().sum()
        else:
            loss_boundary = torch.tensor(0.0) #0.0 (shifts[0, :2] - shifts[-1, :2]).square().sum()

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
    if cls == 0:
        shifts = shifts.detach().cpu()
        shifts[:, 2] -= 0.5

    return shifts.detach().cpu()

class Stats:
    def __init__(self):
        self.iou_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.num_total = 0
        self.num_accurate = np.array([0 for th in iou_thresholds])

    def update(self, ious):
        self.num_total += ious.shape[0]
        for ii, iou_threshold in enumerate(self.iou_thresholds):
            self.num_accurate[ii] += (ious > iou_threshold).float().sum()

    def dump(self):
        print(f'iou acc={num_accurate/num_total}')

class BoxInference:
    def __init__(self):
        self.box_size = {
                         0: torch.tensor([4.8557, 2.1343, 1.7822]),
                         1: torch.tensor([0.9858, 0.8887, 1.7669]),
                         2: torch.tensor([1.8623, 0.8435, 1.7970])
                        }
        self.box_std = {
                        0: torch.tensor([0.9531, 0.2415, 0.3746]),
                        1: torch.tensor([0.1755, 0.1372, 0.1504]),
                        2: torch.tensor([1.8623, 0.8435, 1.7970])
                       }
        self.origin0 = {
                        0: torch.tensor([0, 0, -0.5], dtype=torch.float64),
                        1: torch.tensor([0, 0, 25], dtype=torch.float64),
                        2: torch.tensor([0, 0, 25], dtype=torch.float64),
                       }

    def infer(self, trace, cls, trace_id, args):
        """Infer box attributes given a trace and its class.
        
        """
        # points
        points = trace['points']
        frame_ids = points[:, -1].long()
        min_frame_id = frame_ids.min().item()
        max_frame_id = frame_ids.max().item()
        num_frames = max_frame_id - min_frame_id + 1
        frame_ids -= min_frame_id
        
        # boxes
        gt_boxes = trace['boxes'].numpy()
        gt_boxes = gt_boxes[:, [0,1,2,3,4,5,-1]]
        
        # predicted boxes in frame coordinate system
        pred_boxes = torch.zeros(num_frames, 7)
        pred_boxes[:, 3:6] = self.box_size[cls]
        
        # align box heading directions to smoothed velocity
        centers_in_world = scatter(points[:, :3], frame_ids, reduce='mean',
                                   dim=0, dim_size=num_frames)
        heading_dir_in_world = (centers_in_world[5:] - centers_in_world[:-5]).double()
        heading_dir_in_world = scatter(torch.cat([heading_dir_in_world, heading_dir_in_world], dim=0),
                                       torch.cat(
                                           [torch.arange(num_frames-5),
                                            torch.arange(num_frames-5)+5],
                                           dim=0),
                                       reduce='mean', dim=0, dim_size=num_frames,
                                      )
        heading_dir_in_world = heading_dir_in_world / heading_dir_in_world.norm(p=2, dim=-1).unsqueeze(-1)
        hx, hy = heading_dir_in_world[:, :2].T.numpy()
        heading_angle_in_world = np.arctan2(hy, hx) % (3.1415926 * 2)

        heading_angle_opt = torch.nn.Parameter(torch.tensor(heading_angle_in_world))
        optimizer = torch.optim.Adam([heading_angle_opt], lr=1e-2)
        weight = torch.ones(num_frames, dtype=torch.float64)
        #median_angle = np.median(heading_angle_in_world)
        #median_dir = np.array([np.cos(median_angle), np.sin(median_angle)])
        #median_dir = torch.tensor(median_dir)
        #dist = (heading_dir_in_world[:, :2] - median_dir).square().sum(-1)
        #weight = (-dist).exp()
        #weight[dist > 0.001] = 0.0001
        import ipdb; ipdb.set_trace()
        for itr in range(10000):
            optimizer.zero_grad()
            cos = heading_angle_opt.cos()
            sin = heading_angle_opt.sin()
            hdir = torch.stack([cos, sin], dim=1)

            vec = ((hdir * heading_dir_in_world[:, :2]).sum(-1) - 1.0).square()

            loss0 = (((hdir * heading_dir_in_world[:, :2]).sum(-1) - 1.0).square()*weight).sum()

            velo = hdir[1:] - hdir[:-1]
            loss1 = (velo[1:] - velo[:-1]).square().sum()

            loss = 0.1*loss0 + loss1 
            loss.backward()
            optimizer.step()
            if (itr+1) % 1000 == 0: 
                print(f'itr={itr}, loss={loss.item()}, smooth={loss1.item()}')
        heading_dir_in_world[:, :2] = hdir.detach()
        pred_corners_in_world = box_np_ops.center_to_corner_box3d(centers_in_world.numpy(),
                                                                  pred_boxes[:, 3:6].numpy(),
                                                                  -heading_angle_opt.detach().numpy(),
                                                                  axis=2)
        if args.visualize:
            vis.boxes('box in world', pred_corners_in_world, trace['classes'])

        # load frame transformation to world coordinate system
        transf = torch.zeros(num_frames, 4, 4, dtype=torch.float64)
        for frame_id in range(min_frame_id, max_frame_id+1):
            with open(f'data/Waymo/train/annos/seq_{args.seq_id}_frame_{frame_id}.pkl', 'rb') as fin:
                annos = pickle.load(fin)
            T = annos['veh_to_global'].reshape(4, 4)
            T = torch.tensor(T, dtype=torch.float64)
            transf[frame_id-min_frame_id] = T

        # from frame to world coordinate system
        origins_in_world = (transf[:, :3, :3] @ self.origin0[cls].unsqueeze(0).unsqueeze(-1)
                            ).squeeze(-1) + transf[:, :3, 3]

        # from world coordinate system to frame
        heading_dir_frame = (transf[:, :3, :3].transpose(1, 2) @ heading_dir_in_world.unsqueeze(-1)
                             ).squeeze(-1)
        centers_frame = (transf[:, :3, :3].transpose(1, 2) @ \
                            (centers_in_world - transf[:, :3, 3]).unsqueeze(-1)
                         ).squeeze(-1)
        
        hx, hy = heading_dir_frame[:, :2].T
        heading_angle_frame = np.arctan2(hy, hx)
        
        pred_boxes[:, :3] = centers_frame
        pred_boxes[:, 6] = heading_angle_frame
        pred_boxes = pred_boxes.numpy()
        pred_corners_frame = box_np_ops.center_to_corner_box3d(pred_boxes[:, :3],
                                                               pred_boxes[:, 3:6],
                                                               -pred_boxes[:, -1],
                                                               axis=2)
        pred_corners_frame = torch.tensor(pred_corners_frame, dtype=torch.float64)

        # from frame to world coordinate system
        pred_corners_in_world = (pred_corners_frame @ transf[:, :3, :3].transpose(1, 2)
                                 ) + transf[:, :3, 3].unsqueeze(1)

        gt_corners = trace['corners']
        if args.visualize:
            vis.boxes(f'gt-{trace_id}', gt_corners, trace['classes'])
        pred_boxes = torch.tensor(pred_boxes)
        if args.visualize:
            vis.boxes(f'pred-{trace_id}', pred_corners_in_world, trace['classes'], enabled=False)
        # shift away from origin
        shifts = away_from_origin(points, pred_corners_in_world, origins_in_world, trace['classes'], cls)
        pred_corners_in_world += shifts.unsqueeze(1)
        
        for frame_id in range(min_frame_id, max_frame_id+1):
            shift = shifts[frame_id - min_frame_id]
            T = transf[frame_id-min_frame_id]
            shift = shift @ T[:3, :3]
            shifts[frame_id - min_frame_id] = shift
            pred_boxes[frame_id - min_frame_id, :3] += shift

        ious = torch.diag(iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes.cuda().float(), torch.tensor(gt_boxes).cuda().float()))

        if args.visualize:
            vis.boxes(f'shifted-pred-{trace_id}', pred_corners_in_world, trace['classes'], enabled=True)
            ps_p = vis.pointcloud(f'points-{trace_id}', trace['points'][:, :3], radius=2e-3)
            from_world_origins = points[:, :3] - origins_in_world[frame_ids]
            ps_p.add_vector_quantity('from origin', from_world_origins)
            vis.trace('world origins', origins_in_world)
            vis.show()

        return pred_corners_in_world, gt_corners, pred_boxes


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    box_inference = BoxInference()

    trace_files = glob.glob(f'data/Waymo/train/traces2/seq_{args.seq_id:03d}_trace_*.pt')
    trace_files = sorted(trace_files)
    #with open('work_dirs/waymo_trace_classifer_training/prediction.pkl', 'rb') as fin:
    #    prediction = pickle.load(fin)
        
    corners_, classes_ = [], []
    gt_corners_ = []
    points_, boxes_ = [], []
    for i, trace_file in enumerate(trace_files):
        print(trace_file)
        if i % args.gpus != args.split:
            continue
        token = trace_file.split('/')[-1].split('.')[0]
        trace = torch.load(trace_file)
        trace_id = int(trace_file.split('/')[-1].split('.')[0].split('_')[-1])
        #pred = prediction[token]
        gt_cls = trace['gt_cls']
        if gt_cls == 3:
            continue
        corners, gt_corners, boxes = box_inference.infer(trace, gt_cls.long().item(), trace_id, args)
        box_dict = dict(corners=corners, gt_corners=gt_corners, boxes=boxes)
        torch.save(box_dict, trace_file.replace('traces2', 'boxes').replace('trace', 'box'))
        points_.append(trace['points'])
        corners_.append(corners)
        gt_corners_.append(gt_corners)
        classes_.append(trace['classes'])
        boxes_.append(boxes)

    if args.visualize:
        #boxes_ = torch.cat(boxes_, dim=0)
        #points_ = torch.cat(points_, dim=0)
        #corners_ = torch.cat(corners_, dim=0)
        #gt_corners_ = torch.cat(gt_corners_, dim=0)
        #classes_ = torch.cat(classes_, dim=0)
        #vis.pointcloud('points', points_[:, :3])
        #vis.boxes('boxes', corners_, classes_)
        #vis.boxes('GT-boxes', gt_corners_, classes_)
        vis.show()
