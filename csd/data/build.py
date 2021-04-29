import operator

import torch
from detectron2.data.build import get_detection_dataset_dicts, worker_init_reset_seed
from detectron2.data.common import AspectRatioGroupedDataset, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import setup_logger


def build_ss_train_loader(cfg, mapper):
    """Builds a semi-supervised data loader that yields both labeled and unlabeled images.

    Each batch consists of `cfg.SOLVER.IMS_PER_BATCH_LABELED` labeled and
    `cfg.SOLVER.IMS_PER_BATCH_UNLABELED` unlabeled images, which can be modified
    in `csd/config/config.py` or in a custom `configs/*.yaml` config file
    supplied to your training script.

    Note:
        - here data is just loaded but the actual x-flips used in CSD happen
        inside the trainer loop (which is not the most optimal way though)
        - it's assumed that labeled and unlabeled datasets are two distinct
        non-overlapping datasets that can be loaded separately.

    The final object that is returned is a DataLoader with infinite sampling yielding
    a pair of batches with labeled and unlabeled images with the same aspect ratio within batch.
    Specifically, the DataLoader yields a tuple of lists:
    ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip]).
    """
    # TODO: add support for splitting the same dataset e.g. based on supervision %

    # Wrapper for dataset loader to avoid duplicate code
    load_data_dicts = lambda x: get_detection_dataset_dicts(
        x, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, min_keypoints=0, proposal_files=None
    )

    # Load metadata for labeled and unlabeled datasets
    labeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN)
    unlabeled_dataset_dicts = load_data_dicts(cfg.DATASETS.TRAIN_UNLABELED)

    # Map metadata into actual objects (note: data augmentations also take place here)
    labeled_dataset = MapDataset(labeled_dataset_dicts, mapper)
    unlabeled_dataset = MapDataset(unlabeled_dataset_dicts, mapper)

    # Define data samplers
    assert cfg.DATALOADER.SAMPLER_TRAIN == "TrainingSampler", "Unsupported training sampler: {}".format(
        cfg.DATALOADER.SAMPLER_TRAIN
    )
    labeled_sampler = TrainingSampler(len(labeled_dataset))
    unlabeled_sampler = TrainingSampler(len(unlabeled_dataset))

    return build_ss_batch_data_loader(  # Initialize actual dataloaders
        (labeled_dataset, unlabeled_dataset),
        (labeled_sampler, unlabeled_sampler),
        cfg.SOLVER.IMS_PER_BATCH_LABELED,
        cfg.SOLVER.IMS_PER_BATCH_UNLABELED,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


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
        # Note: this infinite loop was not needed in the original detectron2's implementation, however,
        # for me in python3.7.10 without this I kept getting `StopIteration`
        while True:

            def generate_batch(dataset, buckets, batch_size):
                """Wrapper for batch generator; returns bucket_id."""
                for d in dataset:
                    # Dataset is a DataLoader intance that yields (image, image_x_flipped)
                    # both are of the same size, so we can use d[0]
                    w, h = d[0]["width"], d[0]["height"]
                    bucket_id = 0 if w > h else 1
                    buckets[bucket_id].append(d)
                    if len(buckets[bucket_id]) == batch_size:
                        return buckets[bucket_id]
                # Unreachable code, raise an exception if ended up here
                raise RuntimeError("Dataset should be of infinite size due to the sampler")

            labeled_batch = generate_batch(self.labeled_dataset, self._labeled_buckets, self.labeled_batch_size)
            unlabeled_batch = generate_batch(
                self.unlabeled_dataset, self._unlabeled_buckets, self.unlabeled_batch_size
            )

            # Yield ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip])
            yield (labeled_batch[:], unlabeled_batch[:])
            del labeled_batch[:]
            del unlabeled_batch[:]
