from detectron2.config import CfgNode as CN


def add_csd_config(cfg):
    """Adds CSD-specific default configuration"""

    ### Model parameters
    cfg.MODEL.META_ARCHITECTURE = "CSDGeneralizedRCNN"
    cfg.MODEL.ROI_HEADS.NAME = "CSDStandardROIHeads"

    ### Solver parameters
    # Note: with CSD enabled, the "effective" batch size (in terms of memory used) is twice larger as images get flipped
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.IMS_PER_BATCH_LABELED = 4
    cfg.SOLVER.IMS_PER_BATCH_UNLABELED = 4

    # CSD weight scheduling parameters (see their supplementary)
    # Note that here we change the notationn - T0 defines the number of iterations until the weight is zero,
    # T1 and T2 define the absolute number of iterations when to start ramp up and ramp down of the weight,
    # and T defines the target iteration when the weight is expected to finish ramping down (note: it's OK if
    # it's less than `SOLVER.NUM_ITER`)
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_BETA = 1.0  # Base multiplier for CSD weights (not mentioned in the paper)
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0 = 1  # Train for one iteration without CSD loss
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T1 = 5000
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T2 = 2000
    # Note: even though `T` represents the total number of iterations, it's safe to continue training after `T` iters
    cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T = 20000

    ### Dataset parameters
    # Default datasets are VOC07+12 for training and VOC07 for testing
    # Note only VOC and COCO for object detection are currently supported
    # TODO: test COCO; add support for segmentation
    cfg.DATASETS.TRAIN = ("voc_2007_trainval",)
    cfg.DATASETS.TRAIN_UNLABELED = ("voc_2012_trainval",)
    # Only VOC and COCO are currently supported for evaluation; also only a **single** evaluation dataset
    # is supported (for visualization reasons; if you turn it off, multiple datasets should work)
    cfg.DATASETS.TEST = ("voc_2007_test",)

    ### Auxiliary
    # Note: visualizations work only with Wandb and when a **single** dataset is used, when using multiple datasets
    # comment this out or see & modify `CSDGeneralizedRCNN._log_visualization_to_wandb`
    cfg.USE_WANDB = True  # Comment this out if you don't want to use Wandb
    cfg.WANDB_PROJECT_NAME = "csd-detectron2"  # Wandb project name to log the run to
    cfg.WANDB_LOG_FREQ = 30  # Logging frequency, not recommended to set lower than 20
    cfg.VIS_PERIOD = 300  # Plot training results each <> iterations (sends them to Wandb)
    # # images to plot per visualization run "group", i.e. for RPN/ROI plots how many examples to show; 3 nicely fits in Wandb
    cfg.VIS_IMS_PER_GROUP = 3
    cfg.VIS_MAX_PREDS_PER_IM = 40  # Maximum number of bounding boxes per image in visualization
    cfg.VIS_TEST = True  # Visualize outputs during inference as well

    ### Other parameters
    # Note: for the parameters below only the provided values are supported, changing them may (should) break the code;
    # they are put here just for reference
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.LOAD_PROPOSALS = False

    return cfg
