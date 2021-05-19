"""File with helper methods for logging data to Wandb."""

import torch
import wandb
from detectron2.config import global_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.events import get_event_storage


def log_visualization_to_wandb(
    image, gts, predictions, predictions_mode, max_predictions, title_prefix="", title_suffix="", iter_=None
):
    """Logs provided image along with its GTs and predicted bboxes to Wandb.

    Args:
        image: np.ndarray, (H,W,3) RGB image in 0-255 range
        gts: Instances, list of GT bbox instances; must contain `gt_boxes` and
            `gt_classes` attributes (see `d2.data.detection_utils.annotations_to_instances`)
        predictions: Instances, list of predicted bbox instances.
            If `predictions_mode` is `RPN`, it must contain `proposal_boxes` and
            `objectness_logits` attributes (see
            `d2.modeling.proposal_generator.proposal_utils.find_top_rpn_proposals`).
            If `predictions_mode` is `ROI`, it must contain `pred_boxes`, `scores`, and
            `pred_classes` attributes (see
            `d2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference`).
        predictions_mode: str, must begin with either 'RPN' or 'ROI', defines whether
            `predictions` contain bboxes predicted by RPN or ROI heads

    For details on how exactly the bboxes are logged to Wandb see:
    https://docs.wandb.ai/guides/track/log#images-and-overlays.
    """

    # Obtain class_id to class_name mapping
    # Here we make an assumption that only one dataset is used for training, for multiple
    # datasets one would have to implement a different logic here, e.g. pass the dataset name
    # as an argument or infer it from the image (maybe it's actually stored somewhere)
    if predictions_mode.startswith("RPN"):  # For RPN viz provide only "meta-labels"
        class_id_to_label = {1: "object", 2: "proposal"}
    else:  # For ROI viz provide full vocabulary
        class_id_to_label = MetadataCatalog.get(global_cfg.DATASETS.TRAIN[0]).thing_classes[:]
        class_id_to_label = {k: v for k, v in enumerate(class_id_to_label)}  # Convert to dict (Wandb req)

    viz_meta = {  # To store the vizualization metadata
        "predictions": {"box_data": [], "class_labels": class_id_to_label},
        "ground_truth": {"box_data": [], "class_labels": class_id_to_label},
    }

    if gts is not None:  # Append GT bboxes
        gt_bboxes = gts.gt_boxes.tensor
        gt_classes = gts.gt_classes
        for i in range(len(gts)):
            viz_meta["ground_truth"]["box_data"].append(
                bbox_to_wandb_dict(
                    gt_bboxes[i],
                    int(gt_classes[i]) if predictions_mode.startswith("ROI") else 1,  # don't log real class for RPNs
                    class_id_to_label,
                    image_shape=image.shape,
                )
            )

    # Visualize RPN predictions. For format see:
    # `d2.modeling.proposal_generator.proposal_utils.find_top_rpn_proposals`
    if predictions_mode.startswith("RPN"):
        # Limit the number of proposals (sorted by objectness score)
        box_number = min(len(predictions.proposal_boxes), max_predictions)
        bboxes = predictions.proposal_boxes.tensor[:box_number]
        logits = predictions.objectness_logits
        probs = torch.sigmoid(logits)  # Convert logits to probs
        for i in range(len(bboxes)):
            viz_meta["predictions"]["box_data"].append(
                bbox_to_wandb_dict(
                    bboxes[i],
                    2,  # RPN's proposal ID
                    class_id_to_label,
                    scores={"prob": float(probs[i])},
                    image_shape=image.shape,
                )
            )
    else:
        # Limit the number of proposals (sorted by objectness score)
        box_number = min(len(predictions.pred_boxes), max_predictions)
        bboxes = predictions.pred_boxes.tensor[:box_number]
        probs = predictions.scores
        classes = predictions.pred_classes
        for i in range(len(bboxes)):
            viz_meta["predictions"]["box_data"].append(
                bbox_to_wandb_dict(
                    bboxes[i],
                    int(classes[i]),
                    class_id_to_label,
                    scores={"prob": float(probs[i])},
                    image_shape=image.shape,
                )
            )

    if iter_ is None:
        try:  # Get current iteration
            iter_ = get_event_storage().iter
        except:  # There is no iter when in eval mode - set to
            iter_ = 0

    wandb_img = wandb.Image(image, boxes=viz_meta)  # Send to wandb
    wandb.log(
        {
            f"{title_prefix}{predictions_mode}_predictions{title_suffix}": wandb_img,
            "global_step": iter_,
            "step": iter_,
        },
        step=iter_,
    )


def bbox_to_wandb_dict(xyxy_bbox, class_id, class_id_to_label, image_shape, scores=None):
    """Converts provided variables to wandb bbox-visualization format.

    Args:
        xyxy_bbox: a tensor of size (4,) following the XYXY format (x1, y1, x2, y2)
        class_id: int, predicted or actual class for the bbox
        class_id_to_label: dict, mapping from class_id to class name
        image_shape: shape of the image to calculate the scaled coordinates
        scores (Optional): dict, additional values to attach to the bbox
    Returns:
        dict following wandb format. See
        https://docs.wandb.ai/guides/track/log#images-and-overlays and
        https://medium.com/analytics-vidhya/object-localization-with-keras-2f272f79e03c
    """

    caption = class_id_to_label[class_id]
    if scores is not None and "prob" in scores:
        prob = int(round(scores["prob"] * 100))
        caption = f"{caption} {prob}%"
    d = {
        "position": {
            "minX": xyxy_bbox[0].item() / image_shape[1],
            "maxX": xyxy_bbox[2].item() / image_shape[1],
            "minY": xyxy_bbox[1].item() / image_shape[0],
            "maxY": xyxy_bbox[3].item() / image_shape[0],
        },
        "class_id": class_id,
        "box_caption": caption,
    }
    if scores:
        d["scores"] = scores
    return d
