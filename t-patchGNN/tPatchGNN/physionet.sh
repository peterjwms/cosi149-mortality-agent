#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=phy_no_shuffle.out
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:TitanXP:1


module --ignore-cache load "cuda/11.8"

patience=10
gpu=0

for seed in {1..5}
do
    python run_models.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu
done