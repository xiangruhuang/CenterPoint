import numpy as np
import torch
import pickle
from det3d.structures import Frame
import math

def load_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def save_pickle(obj, path):
    with open(path, 'wb') as fout:
        return pickle.dump(obj, fout)

ORI_LABEL = [1, 2, 4]

def save_boxes_as_annos(seq_id, frame_id, boxes, labels, save_path):
    frame = Frame.from_index(seq_id, frame_id, no_points=True)

    num_boxes = boxes.shape[0]
    boxes_9d = np.zeros((num_boxes, 9), dtype=np.float32)
    boxes_9d[:, :6] = boxes[:, :6]
    boxes_9d[:, -1] = boxes[:, -1]
    
    objects = []
    for i in range(num_boxes):
        obj = dict(id=i, label=ORI_LABEL[labels[i].long().item()],
                   box=boxes_9d[i], num_points=100,
                   detection_difficulty_level=0,
                   combined_difficulty_level=0,
                   name='random')
        objects.append(obj)
    
    anno_path = f'data/Waymo/train/annos/seq_{seq_id}_frame_{frame_id}.pkl'
    annos = load_pickle(anno_path)
    annos['objects'] = objects
    save_pickle(annos, save_path)
