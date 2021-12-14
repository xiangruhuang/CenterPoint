#!/bin/bash
start=$1
end=$2
mkdir -p work_dirs/candidate_traces/
mkdir -p logs

for i in `seq ${start} ${end}`; do
  gpu_id=$((i % 8));
  CUDA_VISIBLE_DEVICES=${gpu_id} OMP_NUM_THREADS=6 python tools/parallel_data_iterator.py configs/waymo/motion_mask_compuation/waymo_naive_obj_tracking.py --split ${i} --step 16 > logs/${i}.log 2>&1 &
done
