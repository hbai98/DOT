# Installation 
1. Create a conda virtual environment and install required packages.

```shell
conda create --name Dhuman --file requirements.txt
conda activate Dhuman
git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
pip install -V -e . 
cd ../
git clone https://github.com/164140757/DigitHuman.git
cd DigitHuman
python setup.py develop

```

2. Create data folder under DHuman and link the actual dataset path ($DATA_ROOT)

