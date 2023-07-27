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
export THS_VAL=1e1 # 1e0 | 1e1
export DATA_ROOT=../../dataset/TanksAndTemple # ../../dataset/nerf_synthetic | ../../dataset/TanksAndTemple
export IN_CKPT_ROOT=~/checkpoints/DOT/pln/tt_sh25 # ~/checkpoints/DOT/pln/syn_sh16 | ~/checkpoints/DOT/pln/tt_sh25
export OUT_CKPT_ROOT=checkpoints/DOT/tt_sh25 # checkpoints/DOT/syn_sh16 | checkpoints/DOT/tt_sh25
export SCENE=Ignatius # chari | Ignatius
export CONFIG_FILE=DOT/nerf_sh/config/tt # DOT/nerf_sh/config/blender | DOT/nerf_sh/config/tt
export epochs=100
export sample_every=20
export prune_every=1
export GPUs=1

# DOT
CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.optimization \
    --input $IN_CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $OUT_CKPT_ROOT/$SCENE/dot.npz \
    --thresh_type $THS_TYPE \
    --thresh_val $THS_VAL \
    --num_epochs $epochs \
    --prune_every $prune_every \
    --sample_every $sample_every 
# DOT(R)
CUDA_VISIBLE_DEVICES=$GPUs,
python -m DOT.octree.optimization \
    --input $IN_CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $OUT_CKPT_ROOT/$SCENE/dot_r.npz \
    --thresh_type $THS_TYPE \
    --thresh_val $THS_VAL \
    --num_epochs $epochs \
    --prune_every $prune_every \
    --sample_every $sample_every \
    --recursive_prune

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
