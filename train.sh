#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

conda activate Adnerf

export DATA_ROOT=../../dataset/BlendedMVS
export CKPT_ROOT=../../dataset/BlendedMVS
export SCENE=Character
export CONFIG_FILE=DOT/nerf_sh/config/tt
export OUT_NAME=test.npz
export epochs=100
export sample_every=20
# export postier=false

python -m DOT.nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

python -m DOT.nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

# python -m octree.optimization \
#     --input $pre_dir/tree_post_val_1e-3_weight_10e.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/sample/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --sample_only \ 
#     --sample_every $sample_every \
# 11074519