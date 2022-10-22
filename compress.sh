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
export CKPT_ROOT=/hy-tmp/syn_sh16_octrees
export pre_dir=/hy-tmp/syn_sh16_octrees/oct_drums_sgd1e7_asthre0_1_rad1_4_sample256
export SCENE=drums
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.compression \
    $CKPT_ROOT/oct_drums_sgd1e7_asthre0_1_rad1_4_sample256/tree.npz \
    --out_dir /hy-tmp/checkpoints/orig \
    --overwrite