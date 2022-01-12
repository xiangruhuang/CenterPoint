import numpy as np
import torch
import glob

import pickle
from det3d.core.utils.visualization import Visualizer
from det3d.structures import Sequence
import argparse
from tqdm import tqdm
from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--box', action='store_true')
    parser.add_argument('--seq_id', type=int)
    parser.add_argument('--trace_path', type=str)
    parser.add_argument('--no_points', action='store_true')

    args = parser.parse_args()

    return args

def get_seq_id(info):
    path = info['path']
    seq_id = int(path.split('/')[-1].split('.')[0].split('_')[1])
    return seq_id

if __name__ == '__main__':
    args = parse_args()
    with open('data/Waymo/infos_train_sequences_filter_zero_gt.pkl', 'rb') as fin:
        infos = pickle.load(fin)

    vis = Visualizer([], [])
    import polyscope as ps
    ps.set_ground_plane_mode('shadow_only')
    ps.set_shadow_darkness(0.5)
    if args.sequence:
        #seq = torch.load('seq.pt')

        seq = Sequence(infos[args.seq_id], no_points=args.no_points)
        seq.toglobal()
        
        pib = seq.points_in_box()
        vis.pointcloud('points-in-box', pib[:, :3], color=[0,0,1])
        vis.boxes('all-box', seq.corners(), seq.classes())
        if not args.no_points:
            vis.pointcloud('all-points', seq.points4d()[:, :3], radius=1e-4)
            # get points in moving box

            points_in_moving_boxes = seq.points_in_moving_boxes()
            ps_mp = vis.pointcloud('moving-points', points_in_moving_boxes[:, :3])
            ps_mp.add_scalar_quantity('frame', points_in_moving_boxes[:, -1])

    trace_files = glob.glob(f'{args.trace_path}/seq_{args.seq_id:03d}_trace_*.pt')
    points_ = []
    corners_ = []
    classes_ = []
    for i, trace_file in enumerate(tqdm(trace_files)):
        trace = torch.load(trace_file)
        points = trace['points']
        corners = trace['corners']
        classes = trace['classes']
        if trace['gt_cls'] == 3:
            continue
        trace_id = int(trace_file.split('/')[-1].split('.')[0].split('_')[-1])
        #vis.pointcloud(f'points-{trace_id}', points[:, :3], color=[1, 0, 0])
        #vis.boxes(f'box-{trace_id}', corners, classes)
        points_.append(points[:, :3])
        corners_.append(corners)
        classes_.append(classes)

    corners_ = torch.cat(corners_, dim=0)
    classes_ = torch.cat(classes_, dim=0)
    points_ = torch.cat(points_, dim=0)
    vis.pointcloud('detected', points_, color=[1, 0, 0])
    if args.box:
        vis.boxes('detected', corners_, classes_)

    import ipdb; ipdb.set_trace()
    vis.show()
