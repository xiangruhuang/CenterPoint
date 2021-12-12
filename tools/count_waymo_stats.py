import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

stat_files = glob.glob('work_dirs/stats/seq_*.pt')
ranges = [-1, 0, 5, 10, 100, 1000, 1000000000]
velo_ranges = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 1000000]
angle_ranges = [0, 1, 3, 5, 10, 20, 200]
num_boxes = torch.tensor([0, 0, 0])
num_moving_boxes = torch.tensor([0, 0, 0])
num_unique_moving_objects = torch.tensor([0, 0, 0])
num_unique_objects = torch.tensor([0, 0, 0])
num_points_in_box = [[], [], []]
num_points_in_moving_box = [[], [], []]
velo = [[], [], []]
turning_angle = [[], [], []]
num_isolated = [[], [], []]
for f in stat_files:
    data = torch.load(f)
    num_boxes += torch.tensor(data['num_boxes'])
    num_moving_boxes += torch.tensor(data['num_moving_boxes'])
    num_unique_objects += torch.tensor(data['num_unique_objects'])
    num_unique_moving_objects += torch.tensor(data['num_unique_moving_objects'])
    for cls in range(3):
        num_points_in_box[cls] += data['num_points_in_box'][cls]
        num_points_in_moving_box[cls] += data['num_points_in_moving_box'][cls]
        velo[cls] += data['velo'][cls]
        turning_angle[cls] += data['turning_angle'][cls]
        num_isolated[cls] += data['num_isolated'][cls]

print(f'num boxes = {num_boxes}')
print(f'num moving boxes = {num_moving_boxes}')
print(f'num unique moving objects = {num_unique_moving_objects}')
for cls in range(3):
    num_points_in_box_cls = torch.tensor(num_points_in_box[cls])
    num_points_in_moving_box_cls = torch.tensor(num_points_in_moving_box[cls])
    angle_cls = torch.tensor(turning_angle[cls])
    angle_cls = angle_cls / np.pi * 180.0
    velo_cls = torch.tensor(velo[cls])
    for i, r in enumerate(ranges[1:]):
        l = ranges[i]
        mask = (num_points_in_box_cls > l) & (num_points_in_box_cls <= r)
        ratio = mask.sum() / mask.shape[0]
        print(f'\t # boxes in range ({l}, {r}] = {mask.sum()} ratio = {ratio:.4f}')
        # moving boxes
        mask = (num_points_in_moving_box_cls > l) & (num_points_in_moving_box_cls <= r)
        ratio = mask.sum() / mask.shape[0]
        print(f'\t # moving boxes in range ({l}, {r}] = {mask.sum()} ratio = {ratio:.4f}')

    for i, r in enumerate(angle_ranges[1:]):
        l = angle_ranges[i]
        mask = (angle_cls > l) & (angle_cls <= r)
        ratio = mask.sum() / mask.shape[0]
        print(f'\t # moving boxes in arange ({l}, {r}] = {mask.sum()} ratio = {ratio:.4f}')
    
    for i, r in enumerate(velo_ranges[1:]):
        l = velo_ranges[i]
        mask = (velo_cls > l) & (velo_cls <= r)
        ratio = mask.sum() / mask.shape[0]
        print(f'\t # moving boxes in vrange ({l}, {r}] = {mask.sum()} ratio = {ratio:.4f}')

    num_isolated_cls = torch.tensor(num_isolated[cls])
    mask = num_isolated_cls[:, -1] >= 10
    ratio = mask.sum() / mask.shape[0]
    print(f'Pr(isolated)={ratio}')
#print(f'num unique objects = {num_unique_objects}')
#print(f'num unique moving objects = {num_unique_moving_objects}')
#print(f'num points in box = {num_points_in_box}')
#print(num_isolated[:, 0] / num_isolated[:, 1], num_isolated[:, 1], num_isolated_frames)
