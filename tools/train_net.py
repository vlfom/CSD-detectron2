"""TODO: add docs
"""

from detectron2.config import set_global_cfg, get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from csd.config import add_csd_config
from csd.engine import BaselineTrainer, CSDTrainer


def setup(args):
    """Reads & merges configs and executes default detectron2's setup"""

    cfg = get_cfg()  # Load default config
    add_csd_config(cfg)  # Extend with CSD-specific config
    cfg.merge_from_file(args.config_file)  # Extend with config from specified file
    cfg.merge_from_list(args.opts)  # Extend with config specified in args

    assert (  # Sanity check
        cfg.SOLVER.IMS_PER_BATCH
        == cfg.SOLVER.IMG_PER_BATCH_LABEL + cfg.SOLVER.IMG_PER_BATCH_UNLABEL
    ), "Total number of images per batch must be equal to the sum of labeled and unlabeled images per batch"

    cfg.freeze()
    set_global_cfg(
        cfg
    )  # TODO: do we need this? Hacky feature to enable global config access

    default_setup(cfg, args)

    return cfg


def main(args):
    """Sets up config, instantiates trainer, and uses it to start the training loop"""

    cfg = setup(args)

    if args.eval_only:  # TODO: implement eval mode
        pass

    # Set up the Trainer based on the specified mode
    if cfg.SOLVER.MODE == "STANDARD":  # Without CSD
        trainer_base = BaselineTrainer
    elif cfg.SOLVER.MODE == "CSD":  # With CSD
        trainer_base = CSDTrainer
    else:
        raise ValueError(f"Specified trainer cannot be found: {cfg.SEMISUPNET.Trainer}")

    trainer = trainer_base(cfg)
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
