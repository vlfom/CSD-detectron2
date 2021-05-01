import os
from typing import Any

import torch
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import global_cfg


class CSDDetectionCheckpointer(DetectionCheckpointer):
    """Adds a few lines of code to upload the checkpoint to wandb.

    See `d2.checkpoint.DetectionCheckpointer` and `fvcore.common.Checkpointer` for all details.
    Changes are commented with "CSD: ...".
    """

    def save(self, name: str, **kwargs: Any) -> None:
        """See `d2.checkpoint.DetectionCheckpointer.save`."""
        super.save(name, **kwargs)

        # CSD: upload checkpoint to Wandb
        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        if global_cfg.USE_WANDB:  
            wandb.save(save_file)
