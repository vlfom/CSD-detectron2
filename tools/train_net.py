"""TODO: add docs
"""

import csd.modeling.meta_arch
import csd.modeling.roi_heads
import detectron2.data
import wandb
from csd.config import add_csd_config
from csd.engine import CSDTrainerManager
from detectron2.config import get_cfg, set_global_cfg
from detectron2.engine import default_argument_parser, default_setup, launch


def setup(args):
    """Reads & merges configs and executes default detectron2's setup"""

    cfg = get_cfg()  # Load default config
    add_csd_config(cfg)  # Extend with CSD-specific config
    cfg.merge_from_file(args.config_file)  # Extend with config from specified file
    cfg.merge_from_list(args.opts)  # Extend with config specified in args

    assert (  # Sanity check
        cfg.SOLVER.IMS_PER_BATCH == cfg.SOLVER.IMS_PER_BATCH_LABELED + cfg.SOLVER.IMS_PER_BATCH_UNLABELED
    ), "Total number of images per batch must be equal to the sum of labeled and unlabeled images per batch"

    cfg.freeze()
    set_global_cfg(cfg)  # Not really useful in this project, but kept for compatibility

    default_setup(cfg, args)

    return cfg


def main(args):
    """Sets up config, instantiates trainer, and uses it to start the training loop"""

    cfg = setup(args)

    if cfg.USE_WANDB:
        # Set up wandb (for tracking visualizations)
        wandb.login()
        wandb.init(project=cfg.WANDB_PROJECT_NAME, sync_tensorboard=True, config=cfg)
    else:
        assert cfg.VIS_PERIOD == 0, "Visualizations without Wandb are not supported"

    if args.eval_only:  # TODO: implement eval mode
        raise NotImplementedError()

    # Set up the Trainer. I extend DefaultTrainer for managing the training loop. Effectively it's
    # not a trainer itself - it merely wraps the training loop, controls hooks, loading/saving, etc.
    # The trainer that actually runs the forward and backward passes is `CSDTrainer` stored in
    # `DefaultTrainer._trainer` (see `DefaultTrainer.__init__()`).
    trainer = CSDTrainerManager(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
