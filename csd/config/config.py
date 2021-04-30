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
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.IMS_PER_BATCH_LABELED = 4  # One labeled and three unlabeled images per batch
    cfg.SOLVER.IMS_PER_BATCH_UNLABELED = 4

    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 90000

    # Recommended values for VOC dataset from the paper, see supplementary
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_BETA = 1.0  # Base multiplier for CSD weights (not mentioned in the paper)
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0 = 1  # Train for one iteration without CSD loss
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T1 = 20000
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T2 = 10000
    # Note: even though `T` represents the total number of iterations, it's safe to continue training after `T` iters
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T = 70000

    ### Auxiliary
    # Note: visualizations work only with Wandb and when a **single** dataset is used, when using multiple datasets
    # comment this out or see & modify `CSDGeneralizedRCNN._log_visualization_to_wandb`
    cfg.USE_WANDB = True  # Comment this out if you don't want to use Wandb
    cfg.WANDB_PROJECT_NAME = "csd-detectron2"
    cfg.VIS_PERIOD = 100  # Plot training results each 100 iterations (sends them to Wandb)

    ### Other parameters
    # Note: for the parameters below only the provided values are supported, changing them may (should) break the code;
    # they are put here just for reference
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.LOAD_PROPOSALS = None

    return cfg
