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

experiment_name=mcots/test/thresh_method/weight/init_1/lr_1e0_5e-1_1e5_sample_3e-1_2e-1_5e3_hs_5e-4
config=opt/configs/syn.json
CKPT_DIR=checkpoints/${experiment_name}
data_dir=data/nerf_synthetic/drums
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo Launching experiment ${expriment_name}
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE
# python -m unittest test.test_mcots.TestMCOTS.test_run_a_round
python opt/ad_opt.py -t $CKPT_DIR ${data_dir} -c ${config}  > $NOHUP_FILE 2>&1  
echo DETACH

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