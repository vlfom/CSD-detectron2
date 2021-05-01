import os

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator


def get_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator(s) for a given dataset.

    Code is taken from D2's `tools/plain_train_net.py`, see it for more details.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type == "coco":
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)

    raise NotImplementedError(
        "no Evaluator for the dataset {} with the type {}".format(
            dataset_name, evaluator_type
        )
    )
