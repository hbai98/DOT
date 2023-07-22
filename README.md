# DOT: Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF

By Haotian Bai, Yiqi Lin, Yize Chen, Lin Wang

## Introduction
[ICCV 2023] The explicit neural radiance field (NeRF) has gained considerable interest for its efficient training and fast inference capabilities, making it a promising direction such as virtual reality and gaming. In particular, PlenOctree (POT) [1], an explicit hierarchical multi-scale octree representation, has emerged as a structural and influential framework. However, POT’s fixed structure for direct optimization is sub-optimal as the scene complexity evolves continuously with updates to cached color and density, necessitating refining the sampling distribution to capture signal complexity accordingly. To address this issue, we propose the dynamic PlenOctree (DOT), which adaptively refines the sample distribution to adjust to changing scene complexity. Specifically, DOT proposes a concise yet novel hierarchical feature fusion strategy during the iterative rendering process. Firstly, it identifies the regions of interest through training signals to ensure adaptive and efficient refinement. Next, rather than directly filtering out valueless nodes, DOT introduces the sampling and pruning operations for octrees to aggregate features, enabling rapid parameter learning. Compared with POT, our DOT outperforms it by enhancing visual quality, reducing over **52.58/68.16%** parameters, and providing **1.7/1.9** times FPS for NeRF-synthetic and Tanks & Temples, respectively. 

[1] Yu, Alex, et al. "Plenoctrees for real-time rendering of neural radiance fields." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

This is the official implementation of ["Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF"](https://github.com/164140757/DOT) in PyTorch. Our code is built on [PlenOctree](https://github.com/sxyu/plenoctree). 

The pretrained octree models are availabel at
```
conda create -n dot python=3.8
conda create -n dot_nerfsh python=3.8

conda env create -f environment.yml
conda activate dot
python -m pip install --upgrade pip

dot:
pip3 install --upgrade torch torchvision torchaudio
pip install -r requirements.txt
cd dependencies/svox 
pip install .


nerfsh:
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt


```
Note the error message 
```
DNN library initialization failed. Look at the errors above for more details.
```
maybe an indicator for the insufficient GPU memory for the NeRF-SH model to be launched. 



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

