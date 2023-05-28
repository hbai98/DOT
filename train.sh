#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

conda activate dot
export THS_TYPE=weight
export THS_VAL=1e0
export DATA_ROOT=../../dataset/BlendedMVS
export CKPT_ROOT=checkpoints/BlendedMVS
export SCENE=Character
export CONFIG_FILE=DOT/nerf_sh/config/tt
export OUT_NAME=test.npz
export epochs=100
export sample_every=20
export GPUs=2
# export postier=false

CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \ 

CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \ 

# --is_jaxnerf_ckpt 
python -m DOT.octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz

CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.optimization \
    --input $pre_dir/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/$OUT_NAME \
    --thresh_type $THS_TYPE \
    --thresh_val $THS_VAL \
    --num_epochs $epochs \ 
    --prune_only \ 
    --prune_every $prune_every \
    # --sample_every $sample_every 
# 11074519

# ns-train vanilla-nerf --data ../../dataset/BlendedMVS
