#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

module load anaconda3
module load cuda-11.4
source activate

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
conda activate Adnerf


export DATA_ROOT=/hy-tmp/nerf_synthetic
export CKPT_ROOT=/hy-tmp/checkpoints
export pre_dir=/hy-tmp/syn_sh16_octrees/oct_drums_sgd1e7_asthre0_1_rad1_4_sample256
export SCENE=drums
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.optimization \
    --input $pre_dir/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/tree_opt.npz