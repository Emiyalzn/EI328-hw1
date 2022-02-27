#!/bin/bash
runs=(1 2 3)
num_partitions=(2 3)
modes=('random' 'yaxis' 'yaxis+overlap')
for mode in ${modes[@]}
do
  for num_partition in ${num_partitions[@]}
  do
    for run in ${runs[@]}
    do
      python main.py --lr_1 0.1 --lr_2 0.01 --alpha_1 0.9 --alpha_2 0.9 --n_hid 32 --model mlqp --type minmax --n_epoch 10000 --partition_mode ${mode} --partition_num ${num_partition} --train_mode sequential
    done
  done
done