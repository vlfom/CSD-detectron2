import logging
import operator

import torch
from detectron2.data.build import (build_batch_data_loader,
                                   get_detection_dataset_dicts,
                                   trivial_batch_collator,
                                   worker_init_reset_seed)
from detectron2.data.common import (AspectRatioGroupedDataset, DatasetFromList,
                                    MapDataset)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size


def build_ss_train_loader(cfg, mapper):
    """Builds a semi-supervised data loader that yields both labeled and unlabeled images.

    Each batch consists of `cfg.SOLVER.IMG_PER_BATCH_LABEL` labeled and
    `cfg.SOLVER.IMG_PER_BATCH_UNLABEL` unlabeled images, which can be modified
    in `csd/config/config.py` or in a custom `configs/*.yaml` config file
    supplied to your training script.

    Note:
        - here data is just loaded but the actual x-flips used in CSD happen
        inside the trainer loop (which is not the most optimal way though)
        - it's assumed that labeled and unlabeled datasets are two distinct
        non-overlapping datasets that can be loaded separately.
    """
    # TODO: add support for splitting the same dataset e.g. using supervision %

    # Wrapper for dataset loader to avoid duplicate code
    load_data_dicts = lambda x: get_detection_dataset_dicts(
        x,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Load metadata for labeled and unlabeled datasets
    labeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN.LABELED)
    unlabeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN.UNLABELED)

    # Map metadata into actual objects (note: data augmentations also take place here)
    if mapper is None:
        mapper = DatasetMapper.from_config(cfg, True)
    labeled_dataset = MapDataset(labeled_dataset_dicts, mapper)
    unlabeled_dataset = MapDataset(unlabeled_dataset_dicts, mapper)

    # Boilerplate code
    assert (
        cfg.DATALOADER.SAMPLER_TRAIN == "TrainingSampler"
    ), "Unsupported training sampler: {}".format(cfg.DATALOADER.SAMPLER_TRAIN)
    labeled_sampler = TrainingSampler(len(labeled_dataset))
    unlabeled_sampler = TrainingSampler(len(unlabeled_dataset))

    return build_ss_batch_data_loader(  # Initialize actual dataloaders
        (labeled_dataset, unlabeled_dataset),
        (labeled_sampler, unlabeled_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_ss_batch_data_loader(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    """Instantiates two data loaders based on provided metadata and wraps them into a single loader.

    Code is largely taken from `detectron2.data.build.build_batch_data_loader`.
    """
    world_size = get_world_size()

    # Check that batch sizes are divisible by the #GPUs
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )
    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    # Calculate per-GPU batch sizes
    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    assert aspect_ratio_grouping, "ASPECT_RATIO_GROUPING = False is not supported yet"

    # Wrapper for DataLoader instantiation to avoid duplicate code
    create_data_loader = lambda dataset, sampler: torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict

    label_data_loader = create_data_loader(label_dataset, label_sampler)
    label_data_loader = create_data_loader(unlabel_dataset, unlabel_sampler)

    return AspectRatioGroupedSSDataset(
        (label_data_loader, unlabel_data_loader),
        (batch_size_label, batch_size_unlabel),
    )


class AspectRatioGroupedSSDataset(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        pass
