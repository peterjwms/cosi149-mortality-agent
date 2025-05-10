python run_models.py ^
    --dataset physionet --state 'def' --history 24 ^
    --patience 10 --batch_size 32 --lr 1e-3 ^
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 ^
    --te_dim 10 --node_dim 10 --hid_dim 64 ^
    --outlayer Linear --seed 1 --gpu 0


@REM patience=10
@REM gpu=0

@REM for seed in {1..5}
@REM do
@REM     python run_models.py \
@REM     --dataset physionet --state 'def' --history 24 \
@REM     --patience $patience --batch_size 32 --lr 1e-3 \
@REM     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
@REM     --te_dim 10 --node_dim 10 --hid_dim 64 \
@REM     --outlayer Linear --seed $seed --gpu $gpu
@REM done