import numpy as np
import torch
import glob
#from det3d.core.utils.visualization import Visualizer
from PytorchHashmap.torch_hash import HashTable
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter
import random
import os
import argparse
from det3d.ops.primitives.primitives_cpu import voxelization

def check_trace(points):
    if points.shape[0] == 0:
        return None
    if points.shape[0] > 100000:
        ev, ep = voxelization(points, torch.tensor([0.2, 0.2, 0.2, 1]), False)[0].T
        num_voxels = ev.max().item() + 1
        voxels = scatter(points[ep], ev, reduce='mean', dim=0, dim_size=num_voxels)
        points = voxels
    if points.shape[0] > 100000:
        ev, ep = voxelization(points, torch.tensor([0.4, 0.4, 0.4, 1]), False)[0].T
        num_voxels = ev.max().item() + 1
        voxels = scatter(points[ep], ev, reduce='mean', dim=0, dim_size=num_voxels)
        points = voxels
    if points.shape[0] > 100000:
        ev, ep = voxelization(points, torch.tensor([0.6, 0.6, 0.6, 1]), False)[0].T
        num_voxels = ev.max().item() + 1
        voxels = scatter(points[ep], ev, reduce='mean', dim=0, dim_size=num_voxels)
        points = voxels
    if points.shape[0] > 100000:
        return None
    # check connectivity
    pc_range = torch.cat([points.min(0)[0]-3, points.max(0)[0]+3], dim=0)
    edges_1 = ht.voxel_graph(points, points, voxel_size, pc_range, 1, radius=2.0, max_num_neighbors=128)
    edges_0 = ht.voxel_graph(points, points, voxel_size, pc_range, 0, radius=1.5, max_num_neighbors=128)
    edges = torch.cat([edges_1, edges_0], dim=-1)
    graph = csr_matrix((torch.ones_like(edges[0].cpu()), (edges[0].cpu(), edges[1].cpu())), shape=[points.shape[0], points.shape[0]])
    num_comps, graph_indices = connected_components(graph)
    graph_indices = torch.tensor(graph_indices).long()
    graph_size = torch.tensor([(graph_indices == c).sum() for c in range(num_comps)])
    if num_comps > 1:
        max_comp_id = graph_size.argmax()
        sub_points = points.clone()[graph_indices == max_comp_id]
        del points
        return check_trace(sub_points)
    
    # check frame integrity
    num_points = points.shape[0]
    points_xyz = points[:, :3]
    frame_ids = points[:, -1].long()
    min_frame_id = frame_ids.min().item()
    max_frame_id = frame_ids.max().item()
    num_frames = max_frame_id - min_frame_id + 1
    if num_frames < 10:
        print('less than 10 frames')
        return None
    frame_ids -= min_frame_id
    frame_size = scatter(torch.ones(num_points), frame_ids,
                         dim=0, dim_size=num_frames, reduce='sum')
    if (frame_size == 0).any():
        # having a empty frame in between
        print('failed at integrity test')
        return None

    # remove short frames
    #min_z = scatter(points[:, 2], frame_ids, dim=0, dim_size=num_frames,
    #                reduce='min')
    #max_z = scatter(points[:, 2], frame_ids, dim=0, dim_size=num_frames,
    #                reduce='max')
    #valid_mask = (max_z - min_z > 0.3)
    #if (valid_mask == False).long().sum() > 0:
    #    print(f'num short frame={(valid_mask==False).long().sum()}')
    #    return check_trace(points[valid_mask[frame_ids]])

    # check trace smoothness
    centers = scatter(points[:, :3], frame_ids,
                      reduce='mean', dim_size=num_frames, dim=0)
    lamb, gamma = 1, 0.1
    frame_centers = torch.nn.Parameter(centers)
    #trend = (frame_centers[-1] - frame_centers[0])
    #trend = trend / trend.norm(p=2, dim=-1)
    optimizer = torch.optim.Adam([frame_centers], lr=1e-3)
    for itr in range(2000):
        optimizer.zero_grad()
        distsq = (points_xyz - frame_centers[frame_ids]).square().sum(-1)
        loss_reg = scatter(distsq, frame_ids, reduce='mean', dim=0, dim_size=num_frames).mean()
        velo = frame_centers[1:] - frame_centers[:-1]
        velo_dir = velo / velo.norm(p=2, dim=-1).unsqueeze(-1)
        loss_angle = ((velo_dir[1:] * velo_dir[:-1]).sum(-1) - 1).square().clip(0, 1).mean()
        loss_smooth = velo.square().sum(-1).mean()
        loss = loss_reg + loss_smooth * gamma + loss_angle * lamb
        loss.backward()
        optimizer.step()
        #print(f'loss={loss.item():.4f}, angle={loss_angle.item():.4f}, smooth={loss_smooth.item():.4f}')
    smoothness = velo.norm(p=2, dim=-1).max().item()
    vnorm = velo.norm(p=2, dim=-1)
    frame_angle = torch.zeros(num_frames)
    velo = frame_centers[4:] - frame_centers[:-4]
    velo_dir = velo / velo.norm(p=2, dim=-1).unsqueeze(-1)
    angles = (velo_dir[1:] * velo_dir[:-1]).sum(-1).clip(-1, 1).arccos() / 3.1415926 * 180.0
    angles = scatter(torch.cat([angles, angles, angles, angles], dim=0),
                     torch.cat([torch.arange(num_frames-5),
                                torch.arange(num_frames-5)+1,
                                torch.arange(num_frames-5)+4,
                                torch.arange(num_frames-5)+5,
                                ], dim=0),
                     dim=0, reduce='mean', dim_size=num_frames)
    #angle = angles[vnorm[:-1] > 0.05].max()
    angle = angles.max()

    if smoothness > 10.0:
        print(f'not smooth, smoothness={smoothness}')
        return None

    if angle > 30.0:
        print(f'not straight enough, max_angle={angle}')
        frame_mask = (angles < 30)
        sub_points = points.clone()[frame_mask[frame_ids]]
        del points
        return check_trace(sub_points)
        #vis.trace('center-trace', frame_centers.detach().cpu().numpy(), radius=2e-3)
        #import ipdb; ipdb.set_trace()
        #vis.show()
        #return False
    
    # check planarity
    #for itr in range(10):
    #    weight = torch.ones(points.shape[0])
    #    c = points[torch.randperm(points.shape[0])[0], :3]
    #    for inner_itr in range(10):
    #        diff = points[:, :3] - c
    #        diffppT = (diff.unsqueeze(-1) @ diff.unsqueeze(-2)) * weight.unsqueeze(-1).unsqueeze(-1)
    #        ppT = diffppT.mean(0)
    #        eigvals, eigvecs = torch.linalg.eigh(ppT.double())
    #        normal = eigvecs[:, 0]
    #        dist2 = (diff * normal).sum(-1) ** 2
    #        sigma2 = dist2.median()
    #        weight = sigma2 / (dist2 + sigma2)
    #    
    #    ratio = (dist2 < 0.05**2).float().mean()
    #    if ratio > 0.4:
    #        print('looks like wall')
    #        return None
    #    
    #    #diff = wall_points[w1, :3] - wall_points[w0, :3]
    #    #wall_points = points.clone()
    #    #wall_points[:, -1] = 0

    #    #wall_pc_range = torch.cat([wall_points.min(0)[0] - 3, wall_points.max(0)[0] + 3], dim=0)
    #    #w0, w1 = ht.voxel_graph(wall_points, wall_points, wall_voxel_size, wall_pc_range, 0, radius=2.0, max_num_neighbors=256)
    #    #diff = wall_points[w1, :3] - wall_points[w0, :3]
    #    #diffppT = (diff.unsqueeze(-1) @ diff.unsqueeze(-2))
    #    #ppT = scatter(diffppT.view(-1, 9), w0.cpu(), dim_size=wall_points.shape[0], reduce='mean', dim=0).view(-1, 3, 3)
    #    #eigvals, eigvecs = torch.linalg.eigh(ppT.double())
    #    #normals = eigvecs[:, :, 0]
    #    #ratio = ((eigvals[:, 1] > 0.10) & (eigvals[:, 2] > 0.10) & (eigvals[:, 0] < 0.04)).float().mean()
    #    #if ratio > 0.5:
    #    #    print('looks like wall')
    #    #    return None

    # check traveled distance
    #travel_dist = (frame_centers[0] - frame_centers[-1]).norm(p=2)
    #if travel_dist < 10.0:
    #    print('not moving')
    #    return None

    return points

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int)
parser.add_argument('--gpus', type=int)
args = parser.parse_args()


ht = HashTable(4000000)
voxel_size = torch.tensor([2.0, 2.0, 2.0, 1])

for seq_id in range(798):
    if seq_id % args.gpus != args.split:
        continue
    trace_files = glob.glob(f'data/Waymo/train/traces/seq_{seq_id:03d}_trace_*.pt')
    TP, FP, FN = 0, 0, 0

    for i, trace_file in enumerate(trace_files):
        print(trace_file)
        trace_id = int(trace_file.split('/')[-1].split('.')[0].split('_')[-1])
        save_path = os.path.join('data/Waymo/train/traces2/',
                                 f'seq_{seq_id:03d}_trace_{trace_id:06d}.pt')
        if os.path.exists(save_path):
            continue
        #vis.clear()
        trace = torch.load(trace_file)
        points = trace['points']
        fake_box_frame_ids = points[:, -1].long().unique()
        cls = trace['cls']
        corners = trace['corners']
        classes = trace['classes']
        #ps_p = vis.pointcloud(f'points-cls-{cls}', points[:, :3], radius=2e-3)
        #ps_p.add_scalar_quantity('frame % 2', points[:, -1].long() % 2)
        #ps_p.add_scalar_quantity('frame', points[:, -1].long())
        #vis.boxes(f'boxes', trace['corners'], trace['classes'])
        #vis.trace('center-opt', frame_centers.detach().cpu().numpy())
        res = check_trace(points)
        pred_cls = (res is not None)
        corners = trace['corners']
        box_centers = corners.mean(1)
        box_dist = (box_centers[1:] - box_centers[:-1]).norm(p=2, dim=-1)
        if (box_dist.sum() < 1.0) or (box_dist.max() > 6.0):
            cls = 3
        save_dict = {}
        if pred_cls:
            frame_ids = res[:, -1].long()
            min_frame_id = frame_ids.min()
            max_frame_id = frame_ids.max()
            box_frame_ids = trace['box_frame_ids']
            num_frames = max_frame_id - min_frame_id + 1
            centers = scatter(res[:, :3], frame_ids - min_frame_id, reduce='mean',
                              dim=0, dim_size=num_frames)
            
            box_mask = (fake_box_frame_ids <= max_frame_id) & (fake_box_frame_ids >= min_frame_id)
            boxes = trace['boxes'][box_mask]
            classes = trace['classes'][box_mask]
            corners = trace['corners'][box_mask]
            box_frame_ids = box_frame_ids[box_mask]
            dist = (corners.mean(1) - centers).norm(p=2, dim=-1).mean()
            is_valid = True
            if dist > 3.0:
                is_valid = False
            if not ((classes == classes[0]).all()):
                is_valid = False
            if not is_valid:
                cls = 3
            else:
                cls = classes[0]
            save_dict['boxes'] = boxes
            save_dict['classes'] = classes
            save_dict['corners'] = corners
            save_dict['points'] = res
            save_dict['box_frame_ids'] = box_frame_ids
            save_dict['valid'] = True
        else:
            save_dict['valid'] = False

        gt_cls = (cls != 3)

        if gt_cls:
            if gt_cls == pred_cls:
                TP += 1
            else:
                FN += 1
        else:
            if gt_cls != pred_cls:
                FP += 1
        torch.save(save_dict, save_path)

        prec = TP * 1.0 / (TP + FP + 1e-6)
        coverage = TP * 1.0 / (FN + TP + 1e-6)
        print(f'num_sample={i}, trace={trace_file}, precision={prec}, coverage={coverage}')
        
