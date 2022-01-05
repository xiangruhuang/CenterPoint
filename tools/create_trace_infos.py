import glob
import os
from tqdm import tqdm
import torch
import pickle

def create_trace_infos(path, save_path):
    seq_files = glob.glob(os.path.join(path, 'seq_*.pt'))
    traces = []
    for seq_file in tqdm(seq_files):
        seq_id = int(seq_file.split('/')[-1].split('.')[0].split('_')[-1])
        save_path_0 = os.path.join(
                          save_path,
                          'train',
                          'traces',
                          f'seq_{seq_id:03d}_trace_000000.pt'
                      )
        #if os.path.exists(save_path_0):
        #    continue
        traces_this = torch.load(seq_file)
        for trace_id, trace in enumerate(traces_this):
            points = trace['points']
            is_valid = trace['dist'] < 3.0
            classes = trace['classes']
            if classes.shape[0] == 0:
                continue
            if not (classes == classes[0]).all():
                is_valid = False
            assert classes[0] < 3
            if not is_valid:
                cls = 3
            else:
                cls = classes[0]
            save_dict = dict(points=points, cls=cls,
                             boxes=trace['boxes'],
                             corners=trace['corners'],
                             box_frame_ids=trace['box_frame_ids'],
                             classes=trace['classes'])
            save_path_t = os.path.join(
                              save_path,
                              'train',
                              'traces',
                              f'seq_{seq_id:03d}_trace_{trace_id:06d}.pt'
                          )
            torch.save(save_dict, save_path_t)
            save_dict['path'] = save_path_t
            save_dict['num_boxes'] = points[:, -1].max() - points[:, -1].min() + 1
            save_dict.pop('points')
            save_dict.pop('corners')
            save_dict.pop('boxes')
            save_dict.pop('box_frame_ids')
            save_dict.pop('classes')
            traces.append(save_dict)

    import ipdb; ipdb.set_trace()
    pkl_save_path = os.path.join(save_path, 'infos_train_trace_classifier.pkl')
    with open(pkl_save_path, 'wb') as fout:
        pickle.dump(traces, fout)

if __name__ == '__main__':
    create_trace_infos('work_dirs/candidate_traces', 'data/Waymo/')
