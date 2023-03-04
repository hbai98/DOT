#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

# ./launch.sh <exp_name> <GPU_id> <data_dir> -c configs/custom.json
#  ./launch.sh test 0 ../../data/nerf_synthetic/chair -c configs/syn.json
export scene=chair
export exp=${scene}_no_rec_stepsize_3_tol_5e-2
export GPU=0

echo Launching experiment $exp
echo GPU $GPU

CKPT_DIR=ckpt/$exp
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

CUDA_VISIBLE_DEVICES=$GPU nohup python opt.py ../../data/nerf_synthetic/chair -t $CKPT_DIR -c configs/syn.json > $NOHUP_FILE 2>&1 &
# CUDA_VISIBLE_DEVICES=$2 nohup python -u opt.py -t $CKPT_DIR ${@:3} > $NOHUP_FILE 2>&1 &
# CUDA_VISIBLE_DEVICES=$2 python -u opt.py -t $CKPT_DIR ${@:3} 

echo DETACH
