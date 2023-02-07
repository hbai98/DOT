#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -m g-node04
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

module load anaconda3
module load cuda-11.4
source activate
conda activate Adnerf

cd svox 

python setup.py install
export CUDA_HOME='/user/local/cuda-11.7'
