# Relationship-Spatialization-for-Depth-Estimation
ECCV 2022 "Relationship Spatialization for Depth Estimation"

# System configuration
*Highly recommend using containers like "docker" or "singularity"*
1. [singularity installation](https://docs.sylabs.io/guides/3.0/user-guide/installation.html)
2. create an `relation_install.def` file. Copy the following into the file: (the singularity would definitely save a whole bunch of time on matching different versions of packages)
```
Bootstrap: docker
From: pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
%post
    apt-key del 7fa2af80
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
    apt-get update
    apt-get install -y gcc
    apt-get install -y g++
    apt-get install -y libglib2.0-0
    apt-get install -y libsm6
    apt-get install -y libx11-dev
    apt-get install -y git
    apt-get install -y bash-completion


    pip install yacs
    pip install torchvision==0.10.0
    pip install numpy
    pip install matplotlib
    pip install h5py
    pip install pickle-mixin
    pip install Pillow
    pip install tqdm
    pip install pytorch3d
    pip install regex
    pip install requests
    pip install easydict
    pip install ninja
    pip install cython
    pip install dill
    pip install opencv-python
    pip install pycocotools
    pip install cffi
    pip install scipy
    pip install msgpack
    pip install pyaml
    pip install tensorboardX
    pip install timm
```
3. Create an singularity image file from the `.def` file: `sudo singularity build relation.sif relation_install.def`
4. Enter singularity shell environment: `singularity shell --nv -B xxx/xxx relation.sif`. 
    
    `xxx/xxx` bound directories' path (like data/source code directory) to your singularity shell environment, `--nv` enables graphic cards.
5. create an virtual python enviroment in the singularity shell: `python -m venv ./venv --system-site-packages`.
6. Enter the python virtual environment: `source venv/bin/activate` 
7. compile "graph-rcnn":
```
cd xx/Relationship-Spatialization-for-Depth-Estimation/lib/scene_parser/rcnn
python setup.py build develop
```
8. compile chamfer-master:
```
cd xx/Relationship-Spatialization-for-Depth-Estimation/chamfer-master
python setup.py install
```
------
Cool! Now all environments would be ready. The checklist:
- Create a singularity image by steps:1,2,3
- Create a python virtual environment by step 5
- Compile graph-rcnn by step 7
- compile chamfer-master by step 8

# Data preparation
1. Download [Kitti](http://www.cvlibs.net/datasets/kitti/raw_data.php) and [NYUv2](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing) datasets (The NYUv2 is provided by [BTS](https://github.com/cleinc/bts/tree/master/pytorch)).
2. prepare relationship features: 
- Get into singularity and activate virtual environment
- `python utils/get_relationship_features.py --dataset nyu --data_path xxxx`

# Training

1. Make directories to deposit pretrained models: 
- `mkdir pretrained`->`cd pretrained`
- download pretrained models [here]()
- clone repository `git clone https://github.com/rwightman/gen-efficientnet-pytorch.git`

2. train:
```
singularity shell --nv -B /home/xxy/Documents/github/Relationship-Spatialization-for-Depth-Estimation -B ~/HDD/dataset ~/Documents/singularity-make/relation.sif
source activate relation
source venv/bin/activate
kitti:
    python train.py --data_path /home/xxy/HDD/dataset/kitti_data --filenames_file ./train_test_inputs/kitti_eigen_train_files_with_gt.txt --data_path_eval /home/xxy/HDD/dataset/kitti_data --filenames_file_eval ./train_test_inputs/kitti_eigen_test_files_with_gt.txt --gt_path /home/xxy/HDD/dataset/kitti_depth/--gt_path_eval /home/xxy/HDD/dataset/kitti_depth/ --dataset kitti --max_depth 80 --max_eval_num 10 --algo baseline --print_every 1 --validate_every 1 --do_kb_crop
nyu:
    python train.py --data_path /home/xxy/HDD/dataset/nyuv2_test/ --data_path_eval /home/xxy/HDD/dataset/nyuv2_test/ --dataset nyu --max_depth 10 --max_eval_num 10 --algo baseline --print_every 1 --validate_every 1
```

# Test
```
1. python test.py nyu

2. python test.py kitti
```