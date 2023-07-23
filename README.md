# DOT: Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF

By [Haotian Bai](https://scholar.google.com/citations?hl=en&user=DIy4cA0AAAAJ), Yiqi Lin, Yize Chen, Lin Wang

## Introduction
[ICCV 2023] The explicit neural radiance field (NeRF) has gained considerable interest for its efficient training and fast inference capabilities, making it a promising direction such as virtual reality and gaming. In particular, PlenOctree (POT) [1], an explicit hierarchical multi-scale octree representation, has emerged as a structural and influential framework. However, POT’s fixed structure for direct optimization is sub-optimal as the scene complexity evolves continuously with updates to cached color and density, necessitating refining the sampling distribution to capture signal complexity accordingly. To address this issue, we propose the dynamic PlenOctree (DOT), which adaptively refines the sample distribution to adjust to changing scene complexity. Specifically, DOT proposes a concise yet novel hierarchical feature fusion strategy during the iterative rendering process. Firstly, it identifies the regions of interest through training signals to ensure adaptive and efficient refinement. Next, rather than directly filtering out valueless nodes, DOT introduces the sampling and pruning operations for octrees to aggregate features, enabling rapid parameter learning. Compared with POT, our DOT outperforms it by enhancing visual quality, reducing over **52.58/68.16%** parameters, and providing **1.7/1.9** times FPS for NeRF-synthetic and Tanks & Temples, respectively. 

[1] Yu, Alex, et al. "Plenoctrees for real-time rendering of neural radiance fields." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

This is the official implementation of ["Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF"](https://github.com/164140757/DOT) in PyTorch. Our code is built on [PlenOctree](https://github.com/sxyu/plenoctree). 

## Overview

## Updates

## Results and checkpoints
| Datasets | Backbone | Top1-Loc Acc | Top5-Loc Acc | GT-Known | Top1-Cls Acc | Top5-Cls Acc | Checkpoints & logs|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| NeRF-Synthetic | Deit-S | 76.4 | 91.6 | 96.6 | 78.5 | 94.5 | [Google Drive](https://drive.google.com/drive/folders/1-FranLy5KSttCPK98ZY27TMXuriE9jkj?usp=sharing) 
| ILSVRC | Deit-S | 56.1 | 66.4 | 68.8 | 76.7 | 93.0 | [Google Drive](https://drive.google.com/drive/folders/1-HZBXo_AoK6W5gwRVh4LD8oyGDYrEc8z?usp=sharing) 
## Usage

### Installation

Please refer to [Pytorch](https://pytorch.org/) for customized installation. 

```
conda create -n dot python=3.8
conda activate dot
python -m pip install --upgrade pip
```
Then, 
```
pip install -r requirements.txt
cd dependencies/svox 
pip install .
```



NeRF-SH . Please refer to [Jax](https://github.com/google/jax#installation) for customized installation. 
```
conda create -n dot_nerfsh python=3.8 -y
conda activate dot_nerfsh
pip install -r requirements.txt
```

Note the error message 
```
DNN library initialization failed. Look at the errors above for more details.
```
maybe an indicator for the insufficient GPU memory for the NeRF-SH model to be launched. 

### Dataset
The pretrained octree and NeRF-SH models from [PlenOctree](https://github.com/sxyu/plenoctree) are availabel at ["Google Drive"](https://drive.google.com/drive/folders/1J0lRiDn_wOiLVpCraf6jM7vvCwDr9Dmx).

## Dataset

You can download the pre-processed synthetic and real datasets used in our paper.
Please also cite the original papers if you use any of them in your work.

Dataset | Download Link | Notes on Dataset Split
---|---|---
[Synthetic-NSVF](https://github.com/facebookresearch/NSVF) | [download (.zip)](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip) | 0_\* (training) 1_\* (validation) 2_\* (testing)
[Synthetic-NeRF](https://github.com/bmild/nerf) | [download (.zip)](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip) | 0_\* (training) 1_\* (validation) 2_\* (testing)
[BlendedMVS](https://github.com/YoYo000/BlendedMVS)  | [download (.zip)](https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip) | 0_\* (training) 1_\* (testing)
[Tanks&Temples](https://www.tanksandtemples.org/) | [download (.zip)](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip) | 0_\* (training) 1_\* (testing)


### Training
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

