def add_csd_config(cfg):
    """Adds CSD-specific default configuration"""

    ### Model parameters
    cfg.MODEL.META_ARCHITECTURE = "CSDGeneralizedRCNN"
    cfg.MODEL.ROI_HEADS.NAME = "CSDStandardROIHeads"

    ### Solver parameters
    cfg.SOLVER.IMS_PER_BATCH = 4  # One labeled and three unlabeled images per batch
    cfg.SOLVER.IMS_PER_BATCH_LABEL = 1
    cfg.SOLVER.IMS_PER_BATCH_UNLABEL = 3

    # Recommended values for VOC dataset from the paper, see supplementary
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_BETA = 1  # Base multiplier for CSD weights (not mentioned in the paper)
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T1 = 20000
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T2 = 10000
    # Note: even though `T` represents the total number of iterations, it's safe to continue training for more than `T`
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T = 70000
    cfg.SOLVER.CSD_WARMUP_ITERS = 1  # Train for one iteration without unlabeled data
    # TODO: implement ^

    # Default datasets are VOC07+12 for training and VOC07 for testing
    # Note only VOC and COCO for object detection are currently supported
    # TODO: add support for additional datasets and tasks (segmentation)
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
