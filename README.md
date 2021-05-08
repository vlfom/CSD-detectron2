# CSD: Consistency-Based Semi-Supervised Learning for Object Detection implementation in Detectron2

This repository contains an unofficial implementation of the method described in CSD [paper by Jeong et al](https://papers.nips.cc/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html) based on [Detectron2](https://github.com/facebookresearch/detectron2) framework. It includes implementation for two-stage RFCN object detector only as single-stage detectors were not the focus of my research. It uses [W&B](https://wandb.ai/) to monitor the progress.

# Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Running scripts](#running-scripts)
4. [Results](#results)
5. [Additional notes](#additional-notes)
   - [Overview of the project structure](https://github.com/vlfom/CSD-detectron2#overview-of-the-project-structure)
   - [Wandb](#wandb)
6. [Planned future features](#future-features)
7. [Credits](#credits)

# Overview

The goal of this project was to verify the effectiveness of the CSD method for two-stage object detectors, implement an easily configurable solution, and to learn the D2 framework. After successful implementation, due to the lack of time, the scope of experiments was limited to a *quick, non-extensive* set on *some* datasets. Specifically, (as of now) only two experiments were run:

1. a baseline that uses RFCN model, ResNet50 backbone, and was trained on VOC07 trainval;
2. a CSD version with the same configuration, but CSD regularization enabled for both labeled and unlabeled datasets, where VOC12 trainval was used as unlabeled data.

Both runs were tested on VOC07 test. To monitor the progress better I also implemented logging the results to Wandb including multiple images with both RPN and ROI predictions (see examples in [wandb](#wandb) below).

**The detailed results are reported [below](#results) and in this [Wandb report](https://wandb.ai/vlfom/csd-detectron2/reports/RFCN-vs-CSD-RFCN-on-VOC07--Vmlldzo2NjAwNjI). Using CSD did not bring significant improvement with the configuration I went for, and one should experiment with a larger CSD weight or longer training.**

In [Overview of the project structure](https://github.com/vlfom/CSD-detectron2#overview-of-the-project-structure) you can get an overview of the code. **The core CSD logic (e.g. to copy to your project) is in [`CSDGeneralizedRCNN`'s `forward()`](https://github.com/vlfom/CSD-detectron2/blob/master/csd/modeling/meta_arch/rcnn.py#L23) and [`CSDTrainer`'s `run_step()`](https://github.com/vlfom/CSD-detectron2/blob/master/csd/engine/trainer.py#L130)** (but keep in mind that they may depend on many other functions/classes).

Please note that the current implementation can be used with other datasets and backbones with no problem, and I plan to run some COCO experiments soon.

The author's official source code for RFCN can be found [here](https://github.com/soo89/CSD-RFCN) (also see their [SSD implementation](https://github.com/soo89/CSD-SSD)), but I struggled to make it run and faced multiple memory issues.

# Installation

### Prerequisites

This repository follows D2's requirements which you can find in the next section. Below I just mention the configuration I used for the experiments.

The code was tested on a machine with Ubuntu 18.04, CUDA V11.1.105, cuDNN 7.5.1, 4xV100 GPUs, Python 3.7.10, torch==1.7.1, torchvision==0.8.2, and detectron2==0.4.

I used AWS EC2 for my experiments. To check specifically which AMI (image) and commands I run on the machine, see [this note](https://www.notion.so/vlfom/Configuring-an-AWS-EC2-VM-with-V100-GPUs-for-D2-training-shared-4c8d1487fa324aa08e4881ff3761d121).

### Installing Detectron2
See D2's [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

### Downloading data
Follow [D2's guide](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) on downloading the datasets.

You can also use the script provided inside the `datasets/` folder to download VOC data by running `sh download_voc.sh`. Run this command **inside** the `datasets/` folder.

# Running scripts

### Baseline

*If you get `ModuleNotFoundError: No module named 'csd'` error, you can add `PYTHONPATH=<your_prefix>/CSD-detectron2` to the beginninng of the command.*

To reproduce baseline results (without visualizations), run the command below:
```python
python tools/run_baseline.py --num-gpus 4 --config configs/voc/baseline_L=07_R50RFCN.yaml
```

You could also run a CSD training script with `CSD_WEIGHT_SCHEDULE_RAMP_BETA=0` (or `CSD_WEIGHT_SCHEDULE_RAMP_T0=<some_large_number>`) but its data loader produces x-flips for all images and requires `IMS_PER_BATCH_UNLABELED` to be at least 1, which would slow down the training process significantly. However, to make sure there are no bugs in CSD implementation, I actually tried running the CSD script with the foregoing parameters for 2K iterations and obtained the results similar to the baseline's.

The `run_baseline.py` script is a duplicate of D2's `tools/train_net.py` ([link](https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py)), I only add a few lines of code enable Wandb logging of scalars.

I used a configuration that D2 provides for training on VOC07+12 in [PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml](https://github.com/facebookresearch/detectron2/blob/master/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml) and slightly modified it. [They report](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#cityscapes--pascal-voc-baselines) reaching 51.9 mAP on VOC07 test when training on VOC07+12 trainval.

To speed up the experiments, for this project it was decided to use a model trained **only on VOC07 trainval** and use VOC07 test for testing. For both baseline and CSD experiments the training duration & LRL schedule were the same.

### CSD

Run the command below to train a model with CSD on VOC07 trainval (labeled) and VOC12 (unlabeled):
```python
python tools/run_net.py --num-gpus 4 --config configs/voc/csd_L=07_U=12_R50RFCN.yaml
```

To *resume* the training, you can run the following (note: the current iteration is also checkpointed in D2):
```python
python tools/run_net.py --resume --num-gpus 4 --config configs/voc/csd_L=07_U=12_R50RFCN.yaml MODEL.WEIGHTS output/your_model_weights.pt
```

Note that the configuration file `configs/voc/csd_L=07_U=12_R50RFCN.yaml` extends the baseline's `configs/voc/baseline_L=07_R50RFCN.yaml` so one can compare what exactly was modified.

The details for all the configuration parameters can be found in `csd/config/config.py`. I tried to document most of parameters and after checking the CSD paper and its supplementary you should have no problems with understanding them (also, the code is extensively documented).

It is recommended to experiment with parameters, as I ran only several configurations.

### Evaluation

Run the command below to evaluate your model on VOC07 test dataset:
```python
python tools/run_net.py --eval-only --config configs/voc/csd_L=07_U=12_R50RFCN.yaml MODEL.WEIGHTS output/your_model_weights.pt
```

# Results

Only **a single CSD experiment** was run due to the lack of time with maximum CSD loss' weight of 0.5 (`CSD_WEIGHT_SCHEDULE_RAMP_BETA=0.5` in config). Though the authors used 1.0 in the paper, using it with batch sizes of 8 made the model diverge in several preliminary runs, so I decided to lower it. The detailed config can be found in `configs/voc/csd_L=07_U=12_R50RFCN.yaml`.

The results can be found in **this Wandb [report](https://wandb.ai/vlfom/csd-detectron2/reports/RFCN-vs-CSD-RFCN-on-VOC07--Vmlldzo2NjAwNjI)**, I also made the project & runs public so you can check them in detail as well.

Using the foregoing configuration, the **improvements from CSD regularization are only marginal**. On several occasions CSD-run outperformed baseline-run (e.g. iterations 6K and 10K), however, in the end, both models converged to nearly the same results. An important note to make is that introducing CSD regularization did not harm the model that hints that with the right hyperparameters it could actually help.

Several things should definitely be tried that I left for the future work:
- Both the baseline and CSD experiments seem to have not overfitted on VOC07 during training based on test APs. Also, the mAP of the baseline (46.6%) is very low comparing to SOTA results. Therefore, it is likely that **CSD regularization was not helpful because the model has not started overfitting yet**. Therefore, increasing training duration must be tried; specifically one can try increasing `SOLVER.MAX_ITER` and `SOLVER.STEPS`, I would start with `27000` and `(18000, 24000)` respectively;
- Training using max_csd_weight of 0.5 with batch sizes of 16 was very stable, therefore, I would continue with increasing the CSD weight to 1.0, and then experimenting with changing the weight schedule, e.g. making the `CSD_WEIGHT_SCHEDULE_RAMP_T2` larger;
- Once the models start overfitting and CSD brings improvements, I would experiment with increasing backbone's capacity, i.e. replacing ResNet50 with ResNet101.

# Additional notes

### Overview of the project structure

To better understand the structure of the project, I first recommend checking the official D2's [quickstart guide](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5), and the following blogposts [[1](https://christineai.blog/detectron2-tutorial-i-high-level-structure/), [2](https://christineai.blog/detectron2-tutorial-ii-learning-detectron2-with-structured-graph/), [3](https://christineai.blog/detectron2-tutorial-iii-config-file/)] that describe the detectron2's projects standard structure.

I tried to leave extensive comments in each file so there should be no problem with following the logic once you are comfortable with D2.
The main files to check are:

- `tools/run_net.py` the starting script that loads the configuration, initializes Wandb, creates the trainer manager and the model, and starts the training/evaluation loop;
- `csd/config/config.py` defines the project-specific configuration; note: parameters in this file define default parameters, they are repeated in `configs/voc/csd_L=07_U=12_R50RFCN.yaml` simply for convenience (e.g. running `python tools/run_net.py --num-gpus 4` should give you the same CSD results);
- `csd/engine/trainer.py` contains the implementation of the training loop; `CSDTrainerManager` controls the training process taking care of data loaders/checkpointing/hooks/etc., while `CSDTrainer` runs the actual training loop; as in many other files, I extend D2's default classes such as `DefaultTrainer` and `SimpleTrainer`, overriding some of their methods, modifying only the needed parts;
- `csd/data/build.py` builds the data loaders; it uses images that are loaded and x-flipped as defined in `csd/data/mapper.py`;
- `csd/modeling/meta_arch/rcnn.py` contains the implementation of the forward pass for CSD; specifically, `forward()` implements the standard forward pass along with CSD logic and returns both of the losses (which `CSDTrainer` uses for backpropagation), and `inference()` implements inference logic; note: half of the code there concerns visualizations using Wandb in which I put a lot of effort for seamless monitoring of RPN's and ROI heads' progress;
- `csd/modeling/roi_heads/roi_heads.py` contains several minor modifications of the default `StandardROIHeads` such as returning both predictions and losses during training (which is needed for CSD in `rcnn.py`);
- `csd/utils/events.py` implements an EventWriter that logs scalars to Wandb;
- `csd/checkpoint/detection_checkpoint.py` implements uploading a model's checkpoint to Wandb.

**The core CSD logic (e.g. to copy to your project) is in [`CSDGeneralizedRCNN`'s `forward()`](https://github.com/vlfom/CSD-detectron2/blob/master/csd/modeling/meta_arch/rcnn.py#L23) and [`CSDTrainer`'s `run_step()`](https://github.com/vlfom/CSD-detectron2/blob/master/csd/engine/trainer.py#L130)** (but keep in mind that they may depend on many other functions/classes).

### Wandb

To monitor the training I used [weights and biases](https://wandb.ai/) (Wandb) web platform. Apart from seamless integration with Tensorboard that D2 uses (one can just use `wandb.init(sync_tensorboard=True)` to sync all the metrics), handy reports that can be used to compare experiments and share results, I really liked their visualizations of bounding boxes.

I log both RPN's and ROI heads' predictions each `cfg.VIS_PERIOD` (default=300) iterations on `cfg.VIS_IMS_PER_GROUP` (default=3) images, including up to `cfg.VIS_MAX_PREDS_PER_IM` (default=40) bboxes per image (both for RPN and ROIs).

I log examples for:

1. a set of random training images with GT matching enabled (see [link](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/proposal_generator/rpn.py#L307) and [link](https://github.com/facebookresearch/detectron2/blob/9246ebc3af1c023cfbdae77e5d976edbcf9a2933/detectron2/modeling/matcher.py#L8));
2. a set of training images fixed at the beginning of the run, to monitor the model's progress specifically on them (without GT matching)
3. a set of random testing images during inference each `cfg.TEST.EVAL_PERIOD` iterations.

Below I put some screenshots of example visualizations (note, iterations number in the header of the table is not precise, visualizations are provided just as an example).

| iter 1k | iter ~2k | iter ~3k |
| -- | -- | -- |
| ![image](https://github.com/vlfom/CSD-detectron2/blob/master/teaser/ex1_iter1000.png?raw=true) | ![image](https://github.com/vlfom/CSD-detectron2/blob/master/teaser/ex1_iter2000.png?raw=true) | ![image](https://github.com/vlfom/CSD-detectron2/blob/master/teaser/ex1_iter3000.png?raw=true) |

# Future features

- [ ] Run an experiment with increased CSD loss weight and training duration
- [ ] Test enabling mask RoI head and how CSD affects segmentation performance
- [ ] Test performance on COCO and LVIS datasets
- [x] Add support for splitting a dataset into labeled and unlabeled parts (e.g. using only 1/5% of data as labeled data)

# Credits

I thank the authors of [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher)  and [Few-Shot Object Detection (FsDet)](https://github.com/ucbdrive/few-shot-object-detection) for publicly releasing their code that assisted me with structuring this project.
