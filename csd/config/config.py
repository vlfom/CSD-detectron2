from detectron2.config import CfgNode as CN


def add_csd_config(cfg):
    """Adds CSD-specific default configuration"""

    ### Model parameters
    cfg.MODEL.META_ARCHITECTURE = "CSDGeneralizedRCNN"
    cfg.MODEL.ROI_HEADS.NAME = "CSDStandardROIHeads"

    ### Dataset parameters
    # Default datasets are VOC07+12 for training and VOC07 for testing
    # Note only VOC and COCO for object detection are currently supported
    # TODO: add support for additional datasets and tasks (segmentation)
    cfg.DATASETS.TRAIN = ("voc_2007_trainval",)
    cfg.DATASETS.TRAIN_UNLABELED = ("voc_2012_trainval",)
    cfg.DATASETS.TEST = ("voc_2007_test",)

    ### Solver parameters
    # Note: with CSD enabled, the effective batch size is twice larger as images get flipped
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.IMS_PER_BATCH_LABELED = 1  # One labeled and three unlabeled images per batch
    cfg.SOLVER.IMS_PER_BATCH_UNLABELED = 1

    cfg.SOLVER.BASE_LR = 0.02  # TODO: 0.001 in CSD-RFCN impl
    cfg.SOLVER.STEPS = (60000, 80000)  # TODO: 50K in CSD-RFCN impl
    cfg.SOLVER.MAX_ITER = 90000  # TODO: 100K in CSD-RFCN impl

    # Recommended values for VOC dataset from the paper, see supplementary
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_BETA = 1  # Base multiplier for CSD weights (not mentioned in the paper)
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0 = 1  # Train for one iteration without unlabeled data
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T1 = 20000
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T2 = 10000
    # Note: even though `T` represents the total number of iterations, it's safe to continue training after `T` iters
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T = 70000

    ### Other parameters
    # Note: for the parameters below only the provided values are supported, changing them may break the code;
    # they are put here just for reference
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.LOAD_PROPOSALS = None

    return cfg
