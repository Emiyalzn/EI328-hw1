#!/bin/bash
runs=(1 2 3)
lr_1s=(0.05 0.1 0.5)
lr_2s=(0.005 0.01 0.05)
alphas=(0.0 0.5 0.9)
for lr1 in ${lr_1s[@]}
do
  for run in ${runs[@]}
  do
    python main.py --lr_1 ${lr1} --lr_2 0.01 --alpha_1 0.9 --alpha_2 0.9 --n_hid 32 --model mlqp --type vanilla --n_epoch 10000
  done
done
for lr2 in ${lr_2s[@]}
do
  for run in ${runs[@]}
  do
    python main.py --lr_1 0.1 --lr_2 ${lr2} --alpha_1 0.9 --alpha_2 0.9 --n_hid 32 --model mlqp --type vanilla --n_epoch 10000
  done
done
for alpha in ${alphas[@]}
do
  for run in ${runs[@]}
  do
    python main.py --lr_1 0.1 --lr_2 0.01 --alpha_1 ${alpha} --alpha_2 ${alpha} --n_hid 32 --model mlqp --type vanilla --n_epoch 10000
  done
done