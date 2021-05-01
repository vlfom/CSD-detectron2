import logging
import os
import time
import weakref

import numpy as np
import torch
import wandb
from csd.checkpoint import CSDDetectionCheckpointer
from csd.data import CSDDatasetMapper, TestDatasetMapper, build_ss_train_loader
from csd.engine import CSDEvalHook
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import (DefaultTrainer, SimpleTrainer, TrainerBase,
                               create_ddp_model, hooks)
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluator,
                                   PascalVOCDetectionEvaluator)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules


class CSDTrainerManager(DefaultTrainer):
    """A trainer manager for the semi-supervised learning task using consistency loss based on x-flips.

    Modifications are minimal comparing to D2's `DefaultTrainer`, so see its documentation for more
    details. The only differences are injection of a different trainer `CSDTrainer` along with weight scheduling
    parameters, and a CSD-specific semi-supervised data loader defined in `build_train_loader`.
    """

    def __init__(self, cfg):
        """Initializes the CSDTrainer.

        Most of the code is from `super.__init__()`, the only change is that for `self._trainer`
        the `CSDTrainer` is used and weight scheduling parameters are injected into it, look for
        "CSD: ... " comments.
        """
        TrainerBase.__init__(self)  # CSD: don't call `super`'s init as we are overriding it
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = CSDTrainer(model, data_loader, optimizer)  # CSD: use a CSD-specific trainer
        # CSD: inject weight scheduling parameters into trainer
        (
            self._trainer.solver_csd_beta,
            self._trainer.solver_csd_t0,
            self._trainer.solver_csd_t1,
            self._trainer.solver_csd_t2,
            self._trainer.solver_csd_t,
        ) = (
            cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_BETA,
            cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0,
            cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T1,
            cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T2,
            cfg.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T,
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = CSDDetectionCheckpointer(  # CSD: use custom checkpointer (only few lines are added there)
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """Changes one hook from `DefaultTrainer.build_hooks()` to enable Wandb logging.

        See `DefaultTrainer.build_hooks` for all details, changes are commented with "CSD: ...".
        """

        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.CSDEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))  # CSD: replace eval hook

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        """Defines a data loader to use in the training loop."""
        dataset_mapper = CSDDatasetMapper(cfg, True)
        return build_ss_train_loader(cfg, dataset_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Defines a data loader to use in the testing loop."""
        dataset_mapper = TestDatasetMapper(cfg, True)
        return build_detection_test_loader(cfg, dataset_name, mapper=dataset_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset.

        Modified from D2's example `tools/train_net.py`"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        raise NotImplementedError(
            "No evaluator implementation for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )


class CSDTrainer(SimpleTrainer):
    """The actual trainer that runs the forward and backward passes"""

    def run_step(self):
        """Implements a training iteration for the CSD method."""

        assert self.model.training, "The model must be in the training mode"

        # Get a tuple of labeled and unlabeled instances (with their x-flipped versions)
        # Format: ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip])
        # where first list (batch) is of size `cfg.SOLVER.IMS_PER_BATCH_LABELED` and the latter
        # is of size `cfg.SOLVER.IMS_PER_BATCH_UNLABELED`
        start = time.perf_counter()
        data_labeled, data_unlabeled = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # A boolean that indicates whether CSD loss should be calculated at this iteration
        # See `config.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0`
        use_csd = self.iter >= self.solver_csd_t0

        # Get losses, format (from :meth:`CSDGeneralizedRCNN.forward`):
        # - "loss_cls", "loss_rpn_cls": bbox roi and rpn classification loss
        # - "loss_box_reg", "loss_rpn_loc": bbox roi and rpn localization loss (see :meth:`FastRCNNOutputLayers.losses`)
        # - "sup_csd_loss_cls": CSD consistency loss for classification on labeled data
        # - "sup_csd_loss_box_reg": CSD consistency loss for localization on labeled data
        # - "unsup_csd_loss_cls", "unsup_csd_loss_box_reg": CSD losses on unlabeled data
        loss_dict = self.model(data_labeled, data_unlabeled, use_csd=use_csd)

        self._update_csd_loss_weight()  # CSD weight scheduling (could be a hook though)
        get_event_storage().put_scalar("csd_weight", self.solver_csd_loss_weight)  # Save for monitoring

        # Sum up the supervised losses
        losses_sup = (
            loss_dict["loss_rpn_cls"] + loss_dict["loss_rpn_loc"] + loss_dict["loss_cls"] + loss_dict["loss_box_reg"]
        )
        get_event_storage().put_scalar("total_sup_loss", losses_sup)  # Save for monitoring

        if use_csd:  # Sum up the CSD losses
            losses_csd = (
                loss_dict["sup_csd_loss_cls"]
                + loss_dict["sup_csd_loss_box_reg"]
                + loss_dict["unsup_csd_loss_cls"]
                + loss_dict["unsup_csd_loss_box_reg"]
            )
        else:
            _device = loss_dict["loss_rpn_loc"].device
            losses_csd = torch.zeros(1).to(_device)
        get_event_storage().put_scalar("total_csd_loss", losses_csd.detach().cpu())  # Save for monitoring

        losses = losses_sup + self.solver_csd_loss_weight * losses_csd  # Calculate the total loss

        self.optimizer.zero_grad()
        losses.backward()

        # Log metrics
        self._write_metrics(loss_dict, data_time)

        # Backprop
        self.optimizer.step()

    def _update_csd_loss_weight(self):
        """Controls weight scheduling for the CSD loss: updates the weight coefficient at each iteration."""

        if self.iter < self.solver_csd_t0:
            self.solver_csd_loss_weight = 0
        elif self.iter < self.solver_csd_t1:
            self.solver_csd_loss_weight = (
                np.exp(-5 * np.power((1 - self.iter / self.solver_csd_t1), 2)) * self.solver_csd_beta
            )
        elif self.iter < self.solver_csd_t - self.solver_csd_t2:
            self.solver_csd_loss_weight = self.solver_csd_beta
        else:
            self.solver_csd_loss_weight = (
                np.exp(
                    -12.5
                    * np.power((1 - (self.solver_csd_t - self.iter) / (self.solver_csd_t - self.solver_csd_t2)), 2)
                )
                * self.solver_csd_beta
            )

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """Adds several lines of code to log metrics to Wandb.

        See `SimpleTrainer._write_metrics` for all details. Changes are commented with "CSD: ...".
        """

        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n" f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

                # CSD: log values to Wandb
                try:
                    iter_ = get_event_storage().iter
                except:  # There is no iter when in eval mode - set to 1
                    iter_ = 1
                wandb.log(metrics_dict, step=iter_)
