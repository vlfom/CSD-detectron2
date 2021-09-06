import ast
import operator

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import MetadataCatalog, print_instances_class_histogram
from detectron2.data.build import (build_batch_data_loader,
                                   get_detection_dataset_dicts,
                                   worker_init_reset_seed)
from detectron2.data.common import (AspectRatioGroupedDataset, DatasetFromList,
                                    MapDataset)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, setup_logger
from numpy.random import default_rng


def build_ss_train_loader(cfg, mapper):
    """Builds a semi-supervised data loader that yields both labeled and unlabeled images.

    Data can be loaded in two modes (defined in `cfg.DATASETS.MODE`):
      - "CROSS_DATASET": labeled and unlabeled images come from two disparate datasets, e.g.
      VOCtrain and VOCtest
      - "RANDOM_SPLIT": labeled and unlabeled images come from the same dataset by splitting it
      into the labeled and unlabeled parts
    For more details see `build_ss_datasets()`.

    Each batch consists of `cfg.SOLVER.IMS_PER_BATCH_LABELED` labeled and
    `cfg.SOLVER.IMS_PER_BATCH_UNLABELED` unlabeled images, which can be modified
    in `csd/config/config.py` or in a custom `configs/*.yaml` config file
    supplied to your training script.

    The actual x-flips happen inside `AspectRatioGroupedSSDataset` that is instantiated by
    `build_ss_batch_data_loader`

    The returned tuple contains (1) a tuple of lists with dicts for labeled and unlabeled images
    and (2) a DataLoader with infinite sampling yielding a pair of batches with labeled and unlabeled
    images with the same aspect ratio within batch.

    Specifically, the returned DataLoader yields a tuple of lists:
    ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip]).
    """

    # Load labeled and unlabeled dataset dicts (either use two separate ones or perform a random split)
    labeled_dataset_dicts, unlabeled_dataset_dicts = build_ss_datasets(cfg)

    # Log the datasets sizes
    if comm.is_main_process():
        logger = setup_logger(name=__name__)
        logger.debug(
            "Number of images in the labeled and unlabeled datasets: {}, {}".format(
                len(labeled_dataset_dicts), len(unlabeled_dataset_dicts)
            )
        )

        # Print updated metadata counts
        print_instances_class_histogram(
            labeled_dataset_dicts, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        )

    # Map metadata into actual objects (note: data augmentations also take place here)
    labeled_dataset = MapDataset(labeled_dataset_dicts, mapper)
    unlabeled_dataset = MapDataset(unlabeled_dataset_dicts, mapper)

    # Define data samplers
    assert cfg.DATALOADER.SAMPLER_TRAIN == "TrainingSampler", "Unsupported training sampler: {}".format(
        cfg.DATALOADER.SAMPLER_TRAIN
    )
    labeled_sampler = TrainingSampler(len(labeled_dataset))
    unlabeled_sampler = TrainingSampler(len(unlabeled_dataset))

    return (
        labeled_dataset_dicts,
        unlabeled_dataset_dicts,
    ), build_ss_batch_data_loader(  # Initialize actual dataloaders
        (labeled_dataset, unlabeled_dataset),
        (labeled_sampler, unlabeled_sampler),
        cfg.SOLVER.IMS_PER_BATCH_LABELED,
        cfg.SOLVER.IMS_PER_BATCH_UNLABELED,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_ss_datasets(cfg):
    """Loads dataset(s), splits it into labeled and unlabeled part if needed, and returns both.

    Data can be loaded in two modes (defined in `cfg.DATASETS.MODE`):
      - "CROSS_DATASET": labeled and unlabeled images come from two disparate datasets, e.g.
      VOCtrain and VOCtest
      - "RANDOM_SPLIT": labeled and unlabeled images come from the same dataset by splitting it
      into the labeled and unlabeled parts.

    For "CROSS_DATASET" mode the function simply loads them separately and passes to `build_ss_batch_data_loader`.
    For "RANDOM_SPLIT" mode the function first loads the dataset to split, and uses the following configuration
    parameters to split it (in descdending priority, at least one must defined):
      - option 1: `cfg.DATASETS.SPLIT_PATH` is defined; the function loads the provided file and uses
      the indices inside to split the datasets into labeled and unlabeled subsets;
      - option 2: `cfg.DATASETS.SPLIT_SUP_PERCENT` is defined; generates a random set of indices of the provided size
      as the percentage of the total dataset size, and uses it to select the "labeled" portion of the dataset,
      while other images are treated as "unlabeled";
      - option 2 additional: `cfg.DATASETS.SPLIT_SEED` is **required** to set `np.random.seed`, it is
      needed to make sure that each of GPUs generates exactly the same data split
    """

    # Wrapper for dataset loader to avoid duplicate code
    load_data_dicts = lambda x: get_detection_dataset_dicts(
        x, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, min_keypoints=0, proposal_files=None
    )

    # Load metadata for the labeled dataset
    labeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN)

    if cfg.DATASETS.MODE == "CROSS_DATASET":
        # Load metadata for the unlabeled dataset
        unlabeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN_UNLABELED)
    elif cfg.DATASETS.MODE == "RANDOM_SPLIT":
        labeled_dataset_dicts, unlabeled_dataset_dicts = split_labeled_dataset(labeled_dataset_dicts, cfg)
    else:
        raise NotImplementedError(f"{cfg.DATASETS.MODE} data mode is not supported.")

    return labeled_dataset_dicts, unlabeled_dataset_dicts


def split_labeled_dataset(dataset_dicts, cfg):
    """Splits the labeled dataset into labeled and unlabeled images, and returns dicts for both.

    For that, either loads images' ids from a file, or generates a random split and saves ids
    of labeled subset to the file.
    Whether to use a file is defined in `cfg.DATASETS.SPLIT_USE_PREDEFINED`.
    The file path is defined in `cfg.DATASETS.SPLIT_PATH`.

    Returns: tuple(list[dict], list[dict]).

    See `build_ss_datasets()`'s docs for more details.
    """

    assert (
        cfg.DATASETS.SPLIT_PATH is not None
    ), "cfg.DATASETS.SPLIT_PATH must be defined when using the RANDOM_SPLIT dataset mode"

    labeled_dicts, unlabeled_dicts = [], []  # Select the corresponding dicts
    if cfg.DATASETS.SPLIT_USE_PREDEFINED:  # Load the pre-defined split from file
        with open(cfg.DATASETS.SPLIT_PATH, "r") as f:  # Load ids of the labeled images
            arr_str = f.read()
        labeled_ids = ast.literal_eval(arr_str)
        labeled_ids_set = set(labeled_ids)
        assert len(labeled_ids) > 0, "The list of ids in the cfg.DATASETS.SPLIT_PATH is empty."
        assert len(labeled_ids) == len(
            labeled_ids_set
        ), "The list of ids in the cfg.DATASETS.SPLIT_PATH contains duplicates"

        for d in dataset_dicts:
            if d["image_id"] in labeled_ids_set:
                labeled_dicts.append(d)
            else:
                unlabeled_dicts.append(d)

        assert len(labeled_dicts) == len(
            labeled_ids
        ), "Some of the images in the cfg.DATASETS.SPLIT_PATH were not found in the dataset"
    else:  # Generate a new split and dump it to file
        assert (
            cfg.DATASETS.SPLIT_SUP_PERCENT is not None
        ), "% of data to use as labeled must be specified when `cfg.DATASETS.SPLIT_USE_PREDEFINED` is False"
        assert (
            cfg.DATASETS.SPLIT_SEED is not None
        ), "Random seed must be specified to make sure that GPUs generate the same indices for the labeled subset."

        images_count = len(dataset_dicts)
        cnt_labeled = int(images_count * cfg.DATASETS.SPLIT_SUP_PERCENT / 100)  # Calculate the size of labeled subset
        assert cnt_labeled >= 1, (
            f"Supervision percent provided in cfg.DATASETS.SPLIT_SUP_PERCENT of {cfg.DATASETS.SPLIT_SUP_PERCENT}% "
            f"is too small for the size of data: {images_count}"
        )

        # Generate indices for labeled images
        rand_g = np.random.default_rng(cfg.DATASETS.SPLIT_SEED)  # Set seed
        labeled_idx = set(rand_g.choice(images_count, size=cnt_labeled, replace=False))

        # Sort all images by ids; necessary for reproducibility
        dataset_dicts = sorted(dataset_dicts, key=lambda x: x["image_id"])

        labeled_ids = []  # Save the ids of the labeled split
        for i, d in enumerate(dataset_dicts):
            if i in labeled_idx:
                labeled_dicts.append(d)
                labeled_ids.append(d["image_id"])
            else:
                unlabeled_dicts.append(d)

        if comm.is_main_process():  # Save the new split to the provided file
            with open(cfg.DATASETS.SPLIT_PATH, "w") as f:
                f.write(str(labeled_ids))

    # Return a tuple of lists with list[dict] for labeled and list[dict] for unlabeled splits
    return labeled_dicts, unlabeled_dicts


def build_ss_batch_data_loader(
    dataset, sampler, total_batch_size_label, total_batch_size_unlabel, *, aspect_ratio_grouping=True, num_workers=0
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
    unlabel_data_loader = create_data_loader(unlabel_dataset, unlabel_sampler)

    return AspectRatioGroupedSSDataset(
        (label_data_loader, unlabel_data_loader),
        (batch_size_label, batch_size_unlabel),
    )


class AspectRatioGroupedSSDataset(AspectRatioGroupedDataset):
    """Groups images from datasets by aspect ratios, yields a tuple of instances.

    See `detectron2.data.common.AspectRatioGroupedDataset` for more details.
    """

    def __init__(self, datasets, batch_sizes):
        self.labeled_dataset, self.unlabeled_dataset = datasets
        self.labeled_batch_size, self.unlabeled_batch_size = batch_sizes
        # There are two "buckets" (which could be called batches) for each type of dataset depending
        # on whether w > h or not for each of the images, see `AspectRatioGroupedDataset`.
        # They must be class members and not temporary variables because both buckets are filled
        # at the same time and the first one that gets filled is yielded as a batch; the images in
        # the unfilled bucket remain cached for the next iteration.
        # Note: each bucket stores **pairs** of (image, image_x_flipped)
        self._labeled_buckets, self._unlabeled_buckets = (
            [[] for _ in range(2)],
            [[] for _ in range(2)],
        )

    def __iter__(self):
        # Note: must use two separate iterators instead of e.g. looping through zip(data1, data2). Because of the
        # aspect ratio grouping into two buckets, one dataset may find a batch of same-ratio instances faster
        # than the other. In such scenario, some images from the former may get skipped.

        labeled_d_iter = iter(self.labeled_dataset)
        unlabeled_d_iter = iter(self.unlabeled_dataset)
        while True:

            def generate_batch(d_iter, buckets, batch_size):
                """Wrapper for batch generator; lazily loads images until any bucket is filled."""

                while True:  # Repeat until one of the buckets get filled
                    for bucket_id in [0, 1]:  # Check if some bucket has images enough for the batch
                        if len(buckets[bucket_id]) == batch_size:
                            return buckets[bucket_id]
                    d = next(d_iter)
                    # Dataset is a DataLoader intance that yields (image, image_x_flipped)
                    # both are of the same size, so we can use d[0]
                    w, h = d[0]["width"], d[0]["height"]
                    bucket_id = 0 if w > h else 1
                    buckets[bucket_id].append(d)

                # Unreachable code
                raise RuntimeError("Dataset should be of infinite size due to the sampler")

            labeled_batch = generate_batch(labeled_d_iter, self._labeled_buckets, self.labeled_batch_size)
            unlabeled_batch = generate_batch(unlabeled_d_iter, self._unlabeled_buckets, self.unlabeled_batch_size)

            # Yield ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip])
            yield (labeled_batch[:], unlabeled_batch[:])
            del labeled_batch[:]
            del unlabeled_batch[:]


def build_detection_train_loader(cfg):
    """Builds a data loader for the baseline trainer with support of training on the subset of labeled data only.

    Most of code comes from `d2.data.build.build_detection_train_loader()`, see it for more details.
    """

    # CSD: check config is supported
    assert cfg.DATALOADER.SAMPLER_TRAIN == "TrainingSampler", "Unsupported training sampler: {}".format(
        cfg.DATALOADER.SAMPLER_TRAIN
    )

    # Original code
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    # CSD: subsample the dataset if needed
    dataset = check_subsample_dataset(dataset, cfg)

    if comm.is_main_process():  # Log counts
        logger = setup_logger(name=__name__)
        logger.debug("Number of images in the dataset: {}".format(len(dataset)))
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    # Original code
    mapper = DatasetMapper(cfg, True)

    sampler = TrainingSampler(len(dataset))

    dataset = DatasetFromList(dataset, copy=False)
    dataset = MapDataset(dataset, mapper)
    sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def check_subsample_dataset(dataset_dicts, cfg):
    """Checks if dataset should be subsampled based on configuration, proceeds, and returns it back."""

    assert (
        cfg.DATASETS.MODE is None or cfg.DATASETS.MODE == "RANDOM_SPLIT"
    ), f"{cfg.DATASETS.MODE} data mode is not supported."

    if cfg.DATASETS.MODE == "RANDOM_SPLIT":  # Check if the dataset should be subsampled according to config
        # Reuse CSD code, ignore the unlabeled split
        labeled_dataset_dicts, _ = split_labeled_dataset(dataset_dicts, cfg)

        # Print updated metadata counts
        if comm.is_main_process():
            print_instances_class_histogram(
                labeled_dataset_dicts, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            )

        return labeled_dataset_dicts

    return dataset_dicts
