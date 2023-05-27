#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J       
#BSUB -W 48:00

# nvidia-smi
module load anaconda3
module load cuda-11.4
source activate
conda activate Adnerf
#  bash train_batch.sh | tee log_weight_1e-3_post.txt;
export THS_TYPE=weight
export THS_VAL=1e0

export DATA_ROOT=../../dataset/TanksAndTemple
export CKPT_ROOT=.../../dataset/DOT/tt
export pre_dir=../checkpoints/plenoctree/tt_sh25/Truck/Truck/octrees
# export pre_dir=../checkpoints/DOT
export SCENE=Truck
export CONFIG_FILE=nerf_sh/config/tt
# export OUT_NAME=sample_20_prune_20_val_1e0_weight_100e.npz
export OUT_NAME=prune_10_val_1e0_weight_10e.npz

export epochs=10
export sample_every=20
export prune_every=1
# export postier=false

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --prune_only \ 
#     --prune_every $prune_every \
#     # --sample_every $sample_every 


# export pre_dir=../checkpoints/plenoctree/tt_sh25/Ignatius/Ignatius/octrees
# export SCENE=Ignatius

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --prune_only \ 
#     --prune_every $prune_every \
#     # --sample_every $sample_every 

# export pre_dir=../checkpoints/plenoctree/tt_sh25/Family/Family/octrees
# export SCENE=Family

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --prune_only \ 
#     --prune_every $prune_every \
#     # --sample_every $sample_every 


# export pre_dir=../checkpoints/plenoctree/tt_sh25/Caterpillar/Caterpillar/octrees
# export SCENE=Caterpillar

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --prune_only \ 
#     --prune_every $prune_every \
#     # --sample_every $sample_every 

export pre_dir=../checkpoints/plenoctree/tt_sh25/Barn/Barn/octrees
export SCENE=Barn

python -m octree.optimization \
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