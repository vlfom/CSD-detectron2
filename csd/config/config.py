def add_csd_config(cfg):
    """Adds CSD-specific default configuration"""

    cfg.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0

    cfg.SOLVER.IMS_PER_BATCH = 4  # One labeled and three unlabeled images per batch
    cfg.SOLVER.IMG_PER_BATCH_LABEL = 1
    cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 3
    cfg.SOLVER.WARMUP_ITERS = 1  # Train for one iteration without unlabeled data

    # Default datasets are VOC07+12 for training and VOC07 for testing
    # Note only VOC and COCO for object detection are currently supported
    # TODO: add support for additional tasks and datasets
    cfg.DATASETS.TRAIN.LABELED = ("voc_2007_trainval",)
    cfg.DATASETS.TRAIN.UNLABELED = ("voc_2012_trainval",)
    cfg.DATASETS.TEST = ("voc_2007_test",)

    # Note: for the parameters below only the provided values are supported;
    # they are put here just for reference
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.LOAD_PROPOSALS = None

    return cfg
