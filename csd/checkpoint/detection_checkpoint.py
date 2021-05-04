import os
from typing import Any

import torch
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import global_cfg


class WandbDetectionCheckpointer(DetectionCheckpointer):
    """Adds a few lines of code to upload the checkpoint to wandb.

    See `d2.checkpoint.DetectionCheckpointer` and `fvcore.common.Checkpointer` for all details.
    Changes are commented with "CSD: ...".
    """

    def save(self, name: str, **kwargs: Any) -> None:
        """See `d2.checkpoint.DetectionCheckpointer.save`."""
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

        # CSD: upload checkpoint to Wandb
        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        if global_cfg.USE_WANDB:
            wandb.save(save_file)
