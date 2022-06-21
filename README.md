# AdaptiveNerf

```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install tqdm
pip install -r requirements.txt
```

Training and evaluation on the **NeRF-Synthetic dataset** ([Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)):
```
export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16/
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

python -m nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/
```
