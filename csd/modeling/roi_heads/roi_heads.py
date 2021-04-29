"""
The RoI heads in this module had to be overriden for a single reason - by default,
Detectron2 returns either losses if in training mode or predicted instances if in evaluation
mode. For the CSD loss, however, we need both losses **and** predictions training. For that, only
`return`s in two methods are modified.
"""

from typing import Any, Dict, List, Optional

import torch
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances


@ROI_HEADS_REGISTRY.register()
class CSDStandardROIHeads(StandardROIHeads):
    """Extends `StandardROIHeads`'s with support for no HNM during training and returning raw predictions.

    Code is largely taken from `StandardROIHeads`."""

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        supervised: Any = True,
    ):
        """Applies RoI using given features and proposals.

        Args:
            supervised: defines the type of forward pass during training, if True - a standard
            forward pass is performed, if False - no GT matching (HNM) is performed, and
            RoI raw predictions are returned

        Returns:
            If `self.training=True`, returns Tuple[Tuple[Tensor, Tensor], Dict], where Dict is a dictionary
            of losses, and a tuple of Tensors are:
            - First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.
            - Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
            (see `FastRCNNOutputLayers.forward` for more details)

            If `self.training=False`, returns Tuple[List[Instances], Dict], where Instances is a
            list of predicted instances per image, and Dict is an empty dictionary (kept for
            compatibility).

        The code is largely taken from :meth:`StandardROIHeads.forward`. The only modified lines
        are noted by "CSD: ..." comments.
        """
        del images
        if self.training and supervised:  # CSD: if self.supervised = False, we don't need HNM
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # CSD: get raw predictions along with losses
            predictions, losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            # CSD: return raw predictions (class_prob and bbox_delta) along with losses
            return predictions, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """Forward logic of the box prediction branch.

        The code is taken from :meth:`StandardROIHeads._forward_box`. The only modified line
        is the return statement; look for "CSD: ..." comment.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return predictions, losses  # CSD: add raw predictions to return
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances