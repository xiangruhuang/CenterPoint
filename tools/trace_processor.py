import numpy as np
import torch
import glob
from det3d.core.utils.visualization import Visualizer

seq_files = glob.glob('work_dirs/candidate_traces/*.pt')
vis = Visualizer([], [])
num_traces = 0
num_boxes = 0
correct = 0

for s, seq_file in enumerate(seq_files):
    print(s, len(seq_files))
    traces = torch.load(seq_file, map_location='cpu')
    num_traces += len(traces)
    for i, trace in enumerate(traces):
        num_boxes += trace['corners'].shape[0]
        #points = trace['points']
        #min_frame_id = points[:, -1].min().long().item()
        #max_frame_id = points[:, -1].max().long().item() + 1
        #for frame_id in range(min_frame_id, max_frame_id):
        #    sub_points = points[(points[:, -1] == frame_id)]
        #    if sub_points.shape[0] == 0:
        #        continue
        #    center = sub_points.mean(0)
        
        #vis.pointcloud('points', points[:, :3], radius=2e-3)
        #vis.boxes('box', trace['corners'], trace['classes'], enabled=False)
        #vis.show()

print(f'num traces = {num_traces}, num boxes = {num_boxes}')
