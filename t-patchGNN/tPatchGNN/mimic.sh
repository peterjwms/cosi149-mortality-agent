#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=mimic_sepsis.out
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:TitanXP:1


module --ignore-cache load "cuda/11.8"

patience=10
gpu=0

for seed in {1..1}
do
    python run_models.py \
    --dataset mimic --state 'def' --history 24 \
    --patience $patience --batch_size 8 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu
done

#     python run_models.py --dataset mimic --state 'def' --history 24 --patience $patience --batch_size 8 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --load 84866