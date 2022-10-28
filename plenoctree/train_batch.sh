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

# experiment_name=mcots/test/init
# config=opt/configs/DOT/syn.json

# pre_dir=/hy-tmp/syn_sh16_octrees/oct_drums_sgd1e7_asthre0_1_rad1_4_sample256/tree.npz
# CKPT_DIR=/hy-tmp/checkpoints/${experiment_name}
# data_dir=/hy-tmp/nerf_synthetic/drums
# mkdir -p $CKPT_DIR
# NOHUP_FILE=$CKPT_DIR/log

# echo Launching experiment ${expriment_name}
# echo CKPT $CKPT_DIR
# echo LOGFILE $NOHUP_FILE
# # python -m unittest test.test_mcots.TestMCOTS.test_run_a_round
# python opt/opt.py -t $CKPT_DIR ${data_dir} -c ${config} -p ${pre_dir} > $NOHUP_FILE 2>&1  
# echo DETACH
#  bash train_batch.sh | tee log_weight_1e-3_post.txt;
export THS_TYPE=weight
export THS_VAL=1e-1

export DATA_ROOT=../data/nerf_synthetic
export CKPT_ROOT=../checkpoints/DOT
export pre_dir=../checkpoints/plenoctree/syn_sh16/drums/drums/octrees
export SCENE=drums
export CONFIG_FILE=nerf_sh/config/blender
export OUT_NAME=tree_no_post_val_1e-1_weight_10e.npz
export epochs=10
# export postier=false

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/chair/chair/octrees
# export SCENE=chair

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/ficus/ficus/octrees
# export SCENE=ficus

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     # --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/hotdog/hotdog/octrees
# export SCENE=hotdog

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/lego/lego/octrees
# export SCENE=lego

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 

export pre_dir=../checkpoints/plenoctree/syn_sh16/materials/materials/octrees
export SCENE=materials

python -m octree.optimization \
    --input $pre_dir/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/$OUT_NAME \
    --thresh_type $THS_TYPE \
    --thresh_val $THS_VAL
    --num_epochs $epochs \ 
    # --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/mic/mic/octrees
# export SCENE=mic

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 

# export pre_dir=../checkpoints/plenoctree/syn_sh16/ship/ship/octrees
# export SCENE=ship

# python -m octree.optimization \
#     --input $pre_dir/tree.npz \
#     --config $CONFIG_FILE \
#     --data_dir $DATA_ROOT/$SCENE/ \
#     --output $CKPT_ROOT/$SCENE/$OUT_NAME \
#     --thresh_type $THS_TYPE \
#     --thresh_val $THS_VAL \
#     --num_epochs $epochs \ 
#     --use_postierior 










