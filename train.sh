#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -m g-node02
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

module load anaconda3
source activate
conda activate cu113

export DATA_ROOT=./data/Synthetic_NeRF
export CKPT_ROOT=./checkpoints/syn_sh16/
export SCENE=chair
export CONFIG_FILE=./config/blender

python -m tools.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

python -m tools.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

# dataset='office_home'
# port=3039
# GPUS=1
# lr='5e-6'
# cfg='configs/swin_base.yaml'
# root='swin_loss'

# source='Art'
# target='Clipart'
# log_path="log/${dataset}/${root}/${source}_${target}/${lr}"
# out_path="results/${dataset}/${root}/${source}_${target}/${lr}"

# python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} dist_pmTrans.py --use-checkpoint \
# --source ${source} --target ${target} --dataset ${dataset}  --tag PM --local_rank 0 --batch-size 30 --head_lr_ratio 10 --log ${log_path} --output ${out_path} \
# --cfg ${cfg} 