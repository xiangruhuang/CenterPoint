#!/bin/bash
list_file=$1
mkdir -p data/Waymo/train/traces2/
mkdir -p trace_logs

for line in `cat ${list_file}`; do
  IFS=',' read -r -a array <<< "$line";
  split=${array[0]};
  gpus=${array[1]};
  gpu_id=${array[2]};
  CUDA_VISIBLE_DEVICES=${gpu_id} OMP_NUM_THREADS=6 python tools/trace_analyze.py --split ${split} --gpus ${gpus} > trace_logs/${split}.log 2>&1 &
done
