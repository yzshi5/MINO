python "exp/exp_climate_U.py" \
    --device 'cuda:0' \
    --data_path 'your_path/dataset/weather' \
    --spath 'your_path/saved_models/MINO_T_Climate' \
    --x_dim 3 \
    --query_dims 32 16 \
    --co_domain 1 \
    --radius 0.2 \
    --epochs 480 \
    --batch_size 48 \
    --dim 256 \
    --num_heads 4 \
    --unet_channels 96 \
    --num_res_blocks 1 \
    --num_unet_heads 4 \
    --attention_res '8' \
    --enc_depth 2 \
    --dec_depth 2 \
    --eval 0

