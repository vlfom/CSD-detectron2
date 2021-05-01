from typing import Any, Dict, List, Optional

import torch
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances


@ROI_HEADS_REGISTRY.register()
class CSDStandardROIHeads(StandardROIHeads):
    """Extends `StandardROIHeads`'s with support for disabling HNM during training and returning raw predictions.

    Both features are required for the CSD loss. Code is largely taken from `StandardROIHeads`."""

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
            predictions, losses = self._forward_box(features, proposals, supervised)
            if supervised:  # CSD: calculate losses only for the standard supervised pass
                # Usually the original proposals used by the box head are used by the mask, keypoint
                # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
                # predicted by the box head.
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
            return predictions, losses  # CSD: return both predictions and losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        supervised: Any = True,
    ):
        """Forward logic of the bbox prediction head.

        The code is taken from :meth:`StandardROIHeads._forward_box`. Additional `supervised` arg is added.
        Look for "CSD: ..." comments to find modified lines.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if supervised:  # CSD: calculate predictions and losses for standard supervised pass
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            else:  # CSD: for unsupervised CSD passes no losses can exist (we don't have GTs)
                losses = None
            return predictions, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
