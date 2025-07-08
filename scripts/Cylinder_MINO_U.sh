python "exp/exp_cylinder_U.py" \
    --device 'cuda:0' \
    --data_path 'your_path/dataset/cylinder' \
    --spath 'your_path/saved_models/MINO_U_Cylinder' \
    --x_dim 2 \
    --query_dims 16 16 \
    --co_domain 3 \
    --radius 0.07 \
    --epochs 300 \
    --batch_size 96 \
    --dim 256 \
    --num_heads 4 \
    --unet_channels 64 \
    --num_res_blocks 1 \
    --num_unet_heads 4 \
    --attention_res '8' \
    --enc_depth 2 \
    --dec_depth 2 \
    --eval 0

