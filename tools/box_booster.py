import pickle
import numpy as np
import os
import torch
import argparse
import random
from tqdm import tqdm

from det3d.core.utils.visualization import Visualizer
from det3d.structures import Sequence, Frame
from det3d.core.bbox import box_np_ops
from det3d.ops.iou3d_nms import iou3d_nms_utils
from tools.boxes2annos import save_boxes_as_annos

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=str)
    parser.add_argument('train', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=1)

    args = parser.parse_args()
    return args

def load_pickle(path):
    with open(path, 'rb') as fin:
        res = pickle.load(fin)
    return res

name2label = {'VEHICLE':0, 'PEDESTRIAN': 1, 'CYCLIST': 2}

args = parse_args()

predictions = load_pickle(args.pred)
infos = load_pickle(args.train)[args.worker_id::args.cpus]

for info in tqdm(infos):
    key = info['path'].split('/')[-1]
    gt_boxes = info['gt_boxes']
    gt_names = info['gt_names']
    gt_labels = []
    for gt_name in gt_names:
        gt_label = name2label[gt_name]
        gt_labels.append(gt_label)
    gt_labels = np.array(gt_labels).reshape(-1)

    seq_id = int(key.split('.')[0].split('_')[1])
    frame_id = int(key.split('.')[0].split('_')[-1])

    frame = Frame.from_index(seq_id, frame_id, args.split)
    pred = predictions[key]

    boxes = pred['box3d_lidar']
    pred_classes = pred['label_preds']
    gt_boxes = torch.tensor(gt_boxes)[:, [0,1,2,3,4,5,-1]]
    gt_labels = torch.tensor(gt_labels).long()
    if (gt_boxes.shape[0] > 0) and (boxes.shape[0] > 0):
        ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes.cuda().float(), gt_boxes.cuda().float())
        mask = gt_labels.unsqueeze(0) == pred_classes.unsqueeze(-1)
        iou_max = (ious.cpu() * mask.float()).max(0)[0]
        corres = (ious.cpu() * mask.float()).argmax(0)
        corres = corres[iou_max > 0.6]
        selected_boxes = boxes[corres]
        selected_classes = pred_classes[corres]
    else:
        selected_boxes = np.zeros((0, 7), dtype=np.float32)
        selected_classes = np.zeros((0), dtype=np.int32)

    save_path = f'data/Waymo/train_ssl_it1/annos/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f'{save_path}/seq_{seq_id}_frame_{frame_id}.pkl'

    save_boxes_as_annos(seq_id, frame_id, selected_boxes, selected_classes, save_path)
