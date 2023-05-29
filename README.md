# DOT

```
conda create -n dot python=3.8
conda create -n dot_nerfsh python=3.8
module load cuda-11.4
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
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.htmlx


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


cmake .. -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=all.