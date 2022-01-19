import pickle
import argparse
from det3d.core.utils.visualization import Visualizer
from det3d.structures import Sequence, Frame
from det3d.core.bbox import box_np_ops
import random
import numpy as np
import torch
from det3d.ops.iou3d_nms import iou3d_nms_utils
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=str)
    parser.add_argument('train', type=str)
    parser.add_argument('--split', type=str, default='val')

    args = parser.parse_args()
    return args

def load_pickle(path):
    with open(path, 'rb') as fin:
        res = pickle.load(fin)
    return res

if __name__ == '__main__':
    args = parse_args()

    predictions = load_pickle(args.pred)
    infos = load_pickle(args.train)

    vis = Visualizer([], [])

    random.shuffle(infos)
    name2label = {'VEHICLE':0, 'PEDESTRIAN': 1, 'CYCLIST': 2}

    num_covered = torch.tensor([0, 0, 0], dtype=torch.long)
    num_total = torch.tensor([0, 0, 0], dtype=torch.long)

    with tqdm(infos) as pbar:
        for info in infos:
            key = info['path'].split('/')[-1]
            vis.clear()
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']
            gt_labels = []
            for gt_name in gt_names:
                gt_label = name2label[gt_name]
                gt_labels.append(gt_label)
            gt_labels = np.array(gt_labels).reshape(-1)
            try:
                gt_corners = box_np_ops.center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1], axis=2)
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
            #vis.boxes('train-box', gt_corners, gt_labels)

            seq_id = int(key.split('.')[0].split('_')[1])
            frame_id = int(key.split('.')[0].split('_')[-1])

            frame = Frame.from_index(seq_id, frame_id, args.split)
            #vis.pointcloud('frame', frame.points)

            #vis.boxes('gt-box', frame.corners, frame.classes)
            pred = predictions[key]

            boxes = pred['box3d_lidar']
            pred_classes = pred['label_preds']
            gt_boxes = torch.tensor(gt_boxes)[:, [0,1,2,3,4,5,-1]]
            gt_labels = torch.tensor(gt_labels).long()
            pbar.update(1)
            if gt_boxes.shape[0] == 0:
                continue
            if boxes.shape[0] > 0:
                try:
                    ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes.cuda().float(), gt_boxes.cuda().float())
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    print(e)
                mask = gt_labels.unsqueeze(0) == pred_classes.unsqueeze(-1)
                coverage = ((ious.cpu() * mask.float()).max(0)[0] > 0.6).long()
                for cls in range(3):
                    num_covered[cls] += coverage[gt_labels == cls].sum()
            for cls in range(3):
                num_total[cls] += (gt_labels == cls).long().sum()

            scores = pred['scores']
            boxes = boxes.numpy()
            pbar.set_description(f'coverage = {(num_covered/num_total).tolist()}')
            pred_corners = box_np_ops.center_to_corner_box3d(boxes[:, :3], boxes[:, 3:6], boxes[:, -1], axis=2)

            #for score_threshold in [0.2, 0.3, 0.4, 0.5]:
            #    mask = scores > score_threshold
            #    vis.boxes(f'pred-box-{score_threshold}', pred_corners[mask], pred_classes[mask])
            #vis.show()
