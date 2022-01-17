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
from det3d.structures import Sequence, Frame
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--frame', action='store_true')

    args = parser.parse_args()

    return args

def load_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def save_pickle(obj, path):
    with open(path, 'wb') as fout:
        return pickle.dump(obj, fout)

ORI_LABEL = [1, 2, 4]

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

def convert_box_annos(args):
    trace_files = glob.glob(f'data/Waymo/train/traces2/seq_{args.seq_id:03d}_trace_*.pt')
    box_files = glob.glob(f'data/Waymo/train/boxes/seq_{args.seq_id:03d}_box_*.pt')
    trace_files = sorted(trace_files)
        
    corners_, classes_ = [], []
    gt_corners_ = []
    points_, boxes_ = [], []
    if args.visualize:
        vis = Visualizer([], [])

    boxes_in_frame = {}

    for i, box_file in enumerate(box_files):
        trace_id = int(box_file.split('/')[-1].split('.')[0].split('_')[-1])
        seq_id = args.seq_id
        
        trace_file = f'data/Waymo/train/traces2/seq_{args.seq_id:03d}_trace_{trace_id:06d}.pt'
        trace_dict = torch.load(trace_file)
        box_dict = torch.load(box_file)
        
        box_frame_ids = box_dict['box_frame_ids']
        boxes = box_dict['boxes']

        for box_frame_id, box, cls in zip(box_frame_ids, boxes, box_dict['classes']):
            box_frame_id = box_frame_id.long().item()
            if boxes_in_frame.get(box_frame_id, None) is None:
                boxes_in_frame[box_frame_id] = []
            box_and_label = dict(box=box, label=torch.tensor(cls))
            boxes_in_frame[box_frame_id].append(box_and_label)

    anno_files = glob.glob(f'data/Waymo/train/annos/seq_{args.seq_id}_frame_*.pkl')
    annos = {}
    for anno_file in tqdm(anno_files):
        frame_id = int(anno_file.split('/')[-1].split('.')[0].split('_')[-1])
        annos[frame_id] = load_pickle(anno_file)
        annos[frame_id]['objects'] = []

    for frame_id in tqdm(boxes_in_frame.keys()):
        boxes = [b['box'] for b in boxes_in_frame[frame_id]]
        labels = [b['label'] for b in boxes_in_frame[frame_id]]
        boxes = torch.stack(boxes, dim=0).numpy()
        labels = torch.stack(labels, dim=0).view(-1)
        corners = box_np_ops.center_to_corner_box3d(boxes[:, :3], boxes[:, 3:6],
                                                    boxes[:, -1], axis=2)

        frame = Frame.from_index(args.seq_id, frame_id, 'train', no_points=True)
        anno_path = frame.path.replace('lidar', 'annos')
        annos_f = annos[frame_id]
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            box_9d = np.zeros(9, dtype=np.float32)
            box_9d[:6] = box[:6]
            box_9d[-1] = box[-1]
            label = label.long().item()
            try:
                object_dict = dict(id=i, label=ORI_LABEL[label], box=box_9d, num_points=100,
                                   detection_difficulty_level=0,
                                   combined_difficulty_level=0,
                                   name='random',
                                   )
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(e)
            annos_f['objects'].append(object_dict)
        

        if args.frame:
            frame = Frame.from_index(args.seq_id, frame_id, 'train')
            vis.pointcloud(f'frame-{frame_id}', frame.points[:, :3])
            vis.boxes(f'gt-box-{frame_id}', frame.corners, frame.classes)
        if args.visualize:
            vis.boxes(f'pred-box-{frame_id}', corners, labels)
            vis.show()
    
    for anno_file in tqdm(anno_files):
        frame_id = int(anno_file.split('/')[-1].split('.')[0].split('_')[-1])
        save_anno_path = anno_file.replace('train', 'train_ssl')
        save_pickle(annos[frame_id], save_anno_path)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    #with open('data/Waymo/infos_train_sequences_filter_zero_gt.pkl', 'rb') as fin:
    #    infos = pickle.load(fin)
   
    for i in range(662, 702):
        if i % args.gpus == args.split:
            args.seq_id = i
            convert_box_annos(args)

