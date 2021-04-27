def add_csd_config(cfg):
    """Adds CSD-specific default configuration"""

    cfg.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0

    cfg.SOLVER.IMS_PER_BATCH = 4  # One labeled and three unlabeled images per batch
    cfg.SOLVER.IMG_PER_BATCH_LABEL = 1
    cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 3
    cfg.SOLVER.WARMUP_ITERS = 1  # Train for one iteration without unlabeled data

    # Default datasets are VOC07+12 for training and VOC07 for testing
    cfg.DATASETS.TRAIN.LABELED = ("voc_2007_trainval",)
    cfg.DATASETS.TRAIN.UNLABELED = ("voc_2012_trainval",)
    cfg.DATASETS.TEST = ("voc_2007_test",)

    return cfg