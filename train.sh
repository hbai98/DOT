#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

# conda activate dot
export THS_TYPE=weight
export THS_VAL=1e0
export DATA_ROOT=../../dataset/BlendedMVS
export CKPT_ROOT=checkpoints/BlendedMVS
export SCENE=Statues #Character Fountain Jade Statues
export CONFIG_FILE=DOT/nerf_sh/config/blendmsv
export epochs=100
export sample_every=20
export prune_every=1
export GPUs=3
# export postier=false

conda activate dot_nerfsh
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

# 
CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz

# POT
CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.POT_opt \
    --input $CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/pot.npz \
    --continue_on_decrease

# DOT
CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.optimization \
    --input $CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/dot.npz \
    --thresh_type $THS_TYPE \
    --thresh_val $THS_VAL \
    --num_epochs $epochs \
    --prune_every $prune_every \
    --sample_every $sample_every 
    # --prune_only \ 

python -m DOT.octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/pot.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --write_images $CKPT_ROOT/$SCENE/octrees/pot_rend

python -m DOT.octree.compression \
    $CKPT_ROOT/$SCENE/octrees/dot.npz \
    --out_dir $CKPT_ROOT/$SCENE/octrees/compress \
    --overwrite
# 11074519

# ns-train vanilla-nerf --data ../../dataset/BlendedMVS
