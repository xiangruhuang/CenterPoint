import sys
import numpy as np

with open(sys.argv[1], 'r') as fin:
    lines = [line.strip() for line in fin.readlines()]
    mAP_dict = {'VEHICLE': {}, 'PEDESTRIAN': {}, 'CYCLIST': {}}
    mAPH_dict = {'VEHICLE': {}, 'PEDESTRIAN': {}, 'CYCLIST': {}}
    for line in lines:
        if not line.startswith('OBJECT_TYPE'):
            continue
        level = int(line.split(':')[0].split('_')[-1])
        obj_t = line.split('_')[3]
        if obj_t == 'SIGN':
            continue
        assert level in [1, 2]
        mAP = float(line.split(' ')[2].replace(']', ''))
        mAPH = float(line.split(' ')[4].replace(']', ''))
        mAP_dict[obj_t][level] = mAP
        mAPH_dict[obj_t][level] = mAPH
    results = [0,0,0,0,0,0]
    for key in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        mAPL1 = mAP_dict[key][1]
        mAPL2 = mAP_dict[key][2]
        mAP = (mAPL1 + mAPL2)/2.0
        mAPHL1 = mAPH_dict[key][1]
        mAPHL2 = mAPH_dict[key][2]
        mAPH = (mAPHL1 + mAPHL2)/2.0
        results[0] += mAPL1/3.0
        results[1] += mAPL2/3.0
        results[2] += mAP/3.0
        results[3] += mAPHL1/3.0
        results[4] += mAPHL2/3.0
        results[5] += mAPH/3.0
    
    for key in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        mAPL1 = mAP_dict[key][1]
        mAPL2 = mAP_dict[key][2]
        mAP = (mAPL1 + mAPL2)/2.0
        mAPHL1 = mAPH_dict[key][1]
        mAPHL2 = mAPH_dict[key][2]
        mAPH = (mAPHL1 + mAPHL2)/2.0
        results += [mAPL1, mAPL2, mAP, mAPHL1, mAPHL2, mAPH]
        
    for i, r in enumerate(results):
        if i > 0:
            print(',', end="")
        print(f'{r:.4f}', end="")

    print('') 
