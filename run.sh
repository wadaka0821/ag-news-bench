#!/bin/bash

batch_sizes=("32" "64" "128" "256")
devices=("cuda" "cpu")
seeds=("42" "742" "146" "8905" "9189")

for device in ${devices[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        for seed in ${seeds[@]}; do
            if [ $device = "cpu" ]; then
                python exp.py --num_container=$1 --batch_size=$batch_size --seed=$seed --base_dir=./res
            else
                python exp.py --num_container=$1 --batch_size=$batch_size --seed=$seed --base_dir=./res --cuda
            fi
        done
    done
done