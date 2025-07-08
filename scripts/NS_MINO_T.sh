python "exp/exp_ns_T.py" \
    --device 'cuda:0' \
    --data_path 'your_path/dataset/N_S' \
    --spath 'your_path/saved_models/MINO_T_NS' \
    --x_dim 2 \
    --dims 64 64 \
    --query_dims 16 16 \
    --co_domain 1 \
    --radius 0.07 \
    --epochs 300 \
    --batch_size 96 \
    --dim 256 \
    --num_heads 4 \
    --enc_depth 5 \
    --dec_depth 2 \
    --eval 0

