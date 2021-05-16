import logging
import os
import time
import weakref
from typing import Dict

import numpy as np
import torch
from csd.checkpoint import WandbDetectionCheckpointer
from csd.data import CSDDatasetMapper, TestDatasetMapper, build_ss_train_loader
from csd.utils import WandbWriter
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import (DefaultTrainer, SimpleTrainer, TrainerBase,
                               create_ddp_model)
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.utils import comm
from detectron2.utils.events import (CommonMetricPrinter, JSONWriter,
                                     TensorboardXWriter, get_event_storage)
from detectron2.utils.logger import setup_logger


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
        self.checkpointer = WandbDetectionCheckpointer(  # CSD: use custom checkpointer (only few lines are added there)
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_train_loader(self, cfg):
        """Defines a data loader to use in the training loop."""
        dataset_mapper = CSDDatasetMapper(cfg, True)

        # In addition to dataloader, fetch a list of labeled and unlabeled dicts
        (labeled_dicts, unlabeled_dicts), train_loader = build_ss_train_loader(cfg, dataset_mapper)

        # If in random_split mode, store the filenames of images for future reference
        if cfg.DATASETS.MODE == "RANDOM_SPLIT":
            labeled_ids = [d["image_id"] for d in labeled_dicts]
            unlabeled_ids = [d["image_id"] for d in unlabeled_dicts]
            self._data_split_ids = (labeled_ids, unlabeled_ids)

        return train_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Defines a data loader to use in the testing loop."""
        dataset_mapper = TestDatasetMapper(cfg, False)
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

    def build_writers(self):
        """Extends default writers with a Wandb writer if Wandb logging was enabled.

        See `d2.engine.DefaultTrainer.build_writers` for more details.
        """

        writers = [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

        if self.cfg.USE_WANDB:
            writers.append(WandbWriter())

        return writers


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

        # A boolean that indicates whether CSD loss should be calculated at this iteration, see
        # `config.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0`
        use_csd = self.iter >= self.solver_csd_t0
        self.solver_csd_loss_weight = 0
        if use_csd:
            self._update_csd_loss_weight()  # Call CSD weight scheduling (could be a hook though)

        # Get losses, format (from :meth:`CSDGeneralizedRCNN.forward`):
        # - "loss_cls", "loss_rpn_cls": bbox roi and rpn classification loss
        # - "loss_box_reg", "loss_rpn_loc": bbox roi and rpn localization loss (see :meth:`FastRCNNOutputLayers.losses`)
        # - "sup_csd_loss_cls": CSD consistency loss for classification on labeled data
        # - "sup_csd_loss_box_reg": CSD consistency loss for localization on labeled data
        # - "unsup_csd_loss_cls", "unsup_csd_loss_box_reg": CSD losses on unlabeled data
        loss_dict = self.model(data_labeled, data_unlabeled, use_csd=use_csd)

        losses_sup = (  # Sum up the supervised losses
            loss_dict["loss_rpn_cls"] + loss_dict["loss_rpn_loc"] + loss_dict["loss_cls"] + loss_dict["loss_box_reg"]
        )
        losses_csd = (  # Sum up the CSD losses
            loss_dict["sup_csd_loss_cls"]
            + loss_dict["sup_csd_loss_box_reg"]
            + loss_dict["unsup_csd_loss_cls"]
            + loss_dict["unsup_csd_loss_box_reg"]
        )
        losses = losses_sup + self.solver_csd_loss_weight * losses_csd  # Calculate the total loss

        self.optimizer.zero_grad()
        losses.backward()

        # Log metrics
        self._write_metrics(loss_dict, data_time)

        # Backprop
        self.optimizer.step()

    def _update_csd_loss_weight(self):
        """Controls weight scheduling for the CSD loss: updates the weight coefficient at each iteration.

        See CSD paper abstract for more details.
        """

        if self.iter < self.solver_csd_t0:
            self.solver_csd_loss_weight = 0
        elif self.iter < self.solver_csd_t1:
            self.solver_csd_loss_weight = (
                np.exp(
                    -5
                    * np.power((1 - (self.iter - self.solver_csd_t0) / (self.solver_csd_t1 - self.solver_csd_t0)), 2)
                )
                * self.solver_csd_beta
            )
        elif self.iter < self.solver_csd_t2:
            self.solver_csd_loss_weight = self.solver_csd_beta
        else:
            self.solver_csd_loss_weight = (
                np.exp(
                    -12.5
                    * np.power(
                        (1 - (self.solver_csd_t - self.iter) / (self.solver_csd_t - self.solver_csd_t2)),
                        2,
                    )
                )
                * self.solver_csd_beta
            )

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
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
            total_sup_loss = sum(metrics_dict[k] for k in ["loss_rpn_cls", "loss_rpn_loc", "loss_cls", "loss_box_reg"])
            if not np.isfinite(total_sup_loss):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n" f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_sup_loss)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

            storage.put_scalar("csd_weight", self.solver_csd_loss_weight)  # CSD loss weight

            # Store aggregates
            csd_loss = (
                metrics_dict["sup_csd_loss_cls"]
                + metrics_dict["sup_csd_loss_box_reg"]
                + metrics_dict["unsup_csd_loss_cls"]
                + metrics_dict["unsup_csd_loss_box_reg"]
            ) * self.solver_csd_loss_weight
            storage.put_scalar("total_csd_loss", csd_loss)  # Sum of the CSD losses
            storage.put_scalar(  # Sum of all losses
                "total_all_loss",
                total_sup_loss + csd_loss,
            )
