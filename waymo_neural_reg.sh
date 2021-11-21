#!/bin/bash
start=$1
end=$2
mkdir -p work_dirs/motion_estimation/
mkdir -p logs

for i in `seq ${start} ${end}`; do
  gpu_id=$((i % 8));
  CUDA_VISIBLE_DEVICES=${gpu_id} python tools/parallel_data_iterator.py configs/waymo/motion_mask_compuation/waymo_neural_reg.py --split ${i} > logs/${i}.log 2>&1 &
done
