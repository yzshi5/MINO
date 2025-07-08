python "exp/exp_climate_T.py" \
    --device 'cuda:0' \
    --data_path 'your_path/dataset/weather' \
    --spath 'your_path/saved_models/MINO_T_Climate' \
    --x_dim 3 \
    --query_dims 32 16 \
    --co_domain 1 \
    --radius 0.2 \
    --kernel_length 0.05 \
    --epochs 480 \
    --batch_size 48 \
    --dim 256 \
    --num_heads 4 \
    --enc_depth 5 \
    --dec_depth 2 \
    --eval 0

