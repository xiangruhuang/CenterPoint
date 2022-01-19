import numpy as np
import torch
import glob
import os

import pickle
from det3d.core.utils.visualization import Visualizer
from torch_scatter import scatter
from det3d.ops.iou3d_nms import iou3d_nms_utils 
from det3d.core.bbox import box_np_ops
import math
import argparse
from det3d.structures import Sequence

vis = Visualizer([], [])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)

    args = parser.parse_args()

    return args


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

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    with open('data/Waymo/infos_train_sequences_filter_zero_gt.pkl', 'rb') as fin:
        infos = pickle.load(fin)
    
    trace_files = glob.glob(f'data/Waymo/train/traces2/seq_{args.seq_id:03d}_trace_*.pt')
    trace_files = sorted(trace_files)
    box_files = glob.glob(f'data/Waymo/train/boxes/seq_{args.seq_id:03d}_box_*.pt')
        
    corners_, classes_ = [], []
    gt_corners_ = []
    points_, boxes_ = [], []
    #if os.path.exists(f'{args.seq_id}.pt'):
    #    seq = torch.load(f'{args.seq_id}.pt')
    #else:
    #    seq = Sequence(infos[args.seq_id], no_points=False)
    #    seq.toglobal()

    for i, trace_file in enumerate(trace_files):
        token = trace_file.split('/')[-1].split('.')[0]
        trace = torch.load(trace_file)
        trace_id = int(trace_file.split('/')[-1].split('.')[0].split('_')[-1])
        
        gt_cls = trace['gt_cls']
        if gt_cls == 3:
            continue
        box_path = trace_file.replace('traces2', 'boxes').replace('trace', 'box')
        if not os.path.exists(box_path):
            continue
        box_dict = torch.load(box_path)
        
        points = trace['points']
        corners = box_dict['corners']
        gt_corners = box_dict['gt_corners']
        boxes = box_dict['boxes']
        classes = box_dict['classes']

        points_.append(trace['points'])
        corners_.append(corners)
        gt_corners_.append(gt_corners)
        classes_.append(classes)
        boxes_.append(boxes)
        try:
            vis.boxes(f'box-{trace_id}', corners, classes)
        except Exception as e:
            import ipdb; ipdb.set_trace()


        #box_file = f'data/Waymo/train/boxes/seq_{args.seq_id:03d}_box_{trace_id:06d}.pt'
        #if os.path.exists(box_file):
        #    box = torch.load(box_file)
        #    box_corners = box['corners']
        #    box_classes = box['classes']
        #    vis.boxes(f'pred-box-{trace_id}', box_corners, box_classes)

    #vis.pointcloud('all-points', seq.points4d()[:, :3], radius=1e-4)
    boxes_ = torch.cat(boxes_, dim=0)
    points_ = torch.cat(points_, dim=0)
    corners_ = torch.cat(corners_, dim=0)
    gt_corners_ = torch.cat(gt_corners_, dim=0)
    classes_ = torch.cat(classes_, dim=0)
    vis.pointcloud('points', points_[:, :3])
    #vis.boxes('boxes', corners_, classes_)
    #vis.boxes('GT-boxes', gt_corners_, classes_)
    vis.show()
