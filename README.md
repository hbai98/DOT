# AdaptiveNerf

```
conda env create -f environment.yml
conda activate Adnerf
pip install .
```

Training and evaluation on the **NeRF-Synthetic dataset** ([Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)):

```
experiment_name=test
config=configs/llff.json
CKPT_DIR=checkpoints/${experiment_name}
data_dir=data/Synthetic_NeRF/Drums
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo Launching experiment ${experiment_name}
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

python -u opt/opt.py -t $CKPT_DIR ${data_dir} > $NOHUP_FILE 2>&1 
echo DETACH
```
