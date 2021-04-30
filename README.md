Work in progress.

# CSD: Consistency-based Semi-supervised learning for object Detection implementation in Detectron2

This repository contains an unofficial implementation of the method described in the [CSD paper by Jeong et al](https://papers.nips.cc/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html) based on the [detectron2](https://github.com/facebookresearch/detectron2) framework.

The repository includes implementation of the method for two-stage RFCN object detector only as single-stage detectors were not the focus of my research. The author's official source code can be found [in this repository](https://github.com/soo89/CSD-RFCN) (also see their [SSD implementation](https://github.com/soo89/CSD-SSD)), but I struggled to make it run and faced unexpected memory issues.

# Installation

### Prerequisites

The code was tested on a machine with Ubuntu 18.04, CUDA 11.2, Cudnn 11.4, 4xV100 GPUs, Python 3.6, torch==1.5, torchvision==1.0.0, and detectron2==0.4 (see `requirements.txt`).

### Installing Detectron2
See detectron2's [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

# Downloading data

You can use the script provided inside the `datasets/` folder to download VOC data by running `sh download_voc.sh`. Please make sure that you run this command **inside** `datasets/`, as it's important that downloaded files get extracted there (or feel free to put your data elsewhere and then set env variable `DETECTRON2_DATASETS` so detectron finds it, see [official docs](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)).

# Training

Training default configuration on VOC07 trainval (labeled) and VOC12 (unlabeled):
```python
python tools/train_net.py --num-gpus 1 --config configs/voc/default_VOC.yaml
```

### Evaluation

...


### Testing

...

### Overview of the project structure

To better understand the structure of the project, I first recommend to check the official Detectron2's [quickstart guide](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5), and the following blogposts [[1](https://christineai.blog/detectron2-tutorial-i-high-level-structure/), [2](https://christineai.blog/detectron2-tutorial-ii-learning-detectron2-with-structured-graph/), [3](https://christineai.blog/detectron2-tutorial-iii-config-file/)] that describe the detectron2's projects standard structure.

The main files to check are:

- `tools/*_net.py` are the main entrance points for the project that load the configuration and pass it to training/evaluation/testing loops;
- `csd/engine/trainer.py` contains the training loop and the core logic, its structure follows the [`plain_train_net.py`'s one](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py) from the official detectron2's repository (*note: it was also possible to extend DefaultTrainer instead of writing the training loop explicitly, but we decided not to do so for better readibility*);
- `csd/data/build.py` builds the data loaders (*note: core logic is in `mapper.py`*);
- `csd/solver/build.py` builds the learning rate schedulers (*note: core logic is in `lr_scheduler.py`*);
- `csd/config/config.py` defines the default configuration (*note: parameters in this file define the "generic configuration" common for most experiments, YAML files in `configs/` folder contain its modifications or customizations*);


### Additional implementation

Here I put some notes on where exactly the configuration is loaded, parameters are set up etc. Please see the diagram from [this blogpost](https://christineai.blog/detectron2-tutorial-ii-learning-detectron2-with-structured-graph/) for a clearer understanding.



# Credits

I thank the authors of [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher)  and [Few-Shot Object Detection (FsDet)](https://github.com/ucbdrive/few-shot-object-detection) for publicly releasing their code that assisted me with structuring this project.