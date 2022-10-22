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

conda activate Adnerf

export DATA_ROOT=/hy-tmp/nerf_synthetic
export CKPT_ROOT=/hy-tmp/checkpoints
export pre_dir=/hy-tmp/syn_sh16_octrees/oct_drums_sgd1e7_asthre0_1_rad1_4_sample256
# export pre_dir=/hy-tmp/checkpoints/drums/
export SCENE=drums
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.evaluation \
    --input $pre_dir/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --write_vid $pre_dir/video_orig.mp4
