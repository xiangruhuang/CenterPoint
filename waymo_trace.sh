#!/bin/bash
list_file=$1
mkdir -p work_dirs/candidate_traces/
mkdir -p logs

for line in `cat ${list_file}`; do
  IFS=',' read -r -a array <<< "$line";
  split=${array[0]};
  gpus=${array[1]};
  gpu_id=${array[2]};
  CUDA_VISIBLE_DEVICES=${gpu_id} OMP_NUM_THREADS=6 python tools/parallel_data_iterator.py configs/waymo/motion_mask_compuation/waymo_naive_obj_tracking.py --split ${split} --gpus ${gpus} > logs/${split}.log 2>&1 &
done
