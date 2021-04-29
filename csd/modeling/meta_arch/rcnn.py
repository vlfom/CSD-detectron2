from typing import Any, Dict, List, Tuple

import detectron2.data.transforms as T
import torch
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger


@META_ARCH_REGISTRY.register()
class CSDGeneralizedRCNN(GeneralizedRCNN):
    """Extends `GeneralizedRCNN`'s forward pass with additional logic for CSD."""

    def forward(
        self,
        batched_inputs_labeled: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        batched_inputs_unlabeled: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        calculate_csd: Any = False,
    ):
        """Performs a standard forward pass along with CSD logic and returns resulting losses.

        Args:
            batched_inputs_labeled: a list, batched instances from :class:`csd.data.CSDDatasetMapper`.
                Each item is based on one of the **labeled** images in the dataset.
                Each item in the list is a tuple, where the first value contains the inputs for one
                image and the second value contains their flipped version.
                Each of "the inputs" is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`GeneralizedRCNN.postprocess` for details.

            batched_inputs_unlabeled: a list in a format similar to `batched_inputs_labeled`.
                Each item is based on one of the **unlabeled** images in the dataset.
                Therefore, the dict for each item contains only the `image` key without
                any ground truth instances.
                During the inference can be set to `None`.

            calculate_csd: a boolean that indicates whether to calculate CSD; it is `True` during all
                training iterations apart from few initial ones, where CSD loss is not calculated,
                see `config.SOLVER.CSD_WEIGHT_SCHEDULE_RAMP_T0`

        Returns:
            if in training mode (`self.training = True`):
                losses: a dict with losses aggregated per key; containing the following keys:
                    - "loss_cls": bbox classification loss
                    - "loss_box_reg": bbox regression loss (see :meth:`FastRCNNOutputLayers.losses`)
                    - "csd_loss_cls": CSD consistency loss for bbox classification
                    - "csd_loss_box_reg": CSD consistency loss for bbox regression

        Specifically:
            - first, performs the standard forward pass on labeled input data
            - then, for both labeled and unlabeled inputs:
                - extracts features from the backbone for original and flipped inputs
                - generates RPN proposals for orig. inputs
                - modifies generated RPN proposals to obtain ones for flipped inputs (coords arithmetics)
                - generates class_scores and deltas for each proposal (for both orig. & flipped inputs)
                - discards bboxes where background is the dominant predicted class
                - applies CSD classification and localization loss assuming bboxes are matched
        To better understand the logic in this method, first see :meth:`GeneralizedRCNN.forward`.
        """

        logger = setup_logger(name=__name__)  # TODO: remove all logging from here

        losses = {}  # Placeholder for future loss accumulation

        ### Split labeled & unlabeled inputs and their flipped versions into separate variables
        labeled_inp, labeled_inp_flip = zip(*batched_inputs_labeled)
        unlabeled_inp, unlabeled_inp_flip = zip(*batched_inputs_unlabeled)

        ### If in inference mode, return predictions for labeled inputs (skipping unlabeled batch)
        if not self.training:
            return self.inference(labeled_inp)

        ### Preprocess inputs
        # We need GTs only for labeled inputs, for others - ignore
        labeled_im, labeled_gt = self._preprocess_images_and_get_gt(labeled_inp)
        if calculate_csd:
            labeled_im_flip, _ = self._preprocess_images_and_get_gt(labeled_inp_flip)
            # For unlabeled inputs, no GTs exist - ignore
            unlabeled_im, _ = self._preprocess_images_and_get_gt(unlabeled_inp)
            unlabeled_im_flip, _ = self._preprocess_images_and_get_gt(unlabeled_inp_flip)

        ### Backbone feature extraction
        # Extract features for all images and their flipped versions
        labeled_feat = self.backbone(labeled_im.tensor)
        if calculate_csd:
            labeled_feat_flip = self.backbone(labeled_im_flip.tensor)
            unlabeled_feat = self.backbone(unlabeled_im.tensor)
            unlabeled_feat_flip = self.backbone(unlabeled_im_flip.tensor)

        ### RPN proposals generation
        logger.debug("Generating proposals for labeled")
        # As described in the CSD paper, generate proposals only for non-flipped images
        labeled_prop, labeled_proposal_losses = self.proposal_generator(labeled_im, labeled_feat, labeled_gt)
        losses.update(labeled_proposal_losses)  # Save RPN losses

        if calculate_csd:
            # For unlabeled images there is no GTs which would cause an error inside RPN;
            # however, we use a hack: set `training=False` temporarily and hope that it doesn't crash :)
            # TODO: check that it works
            logger.debug("Generating proposals for unlabeled")
            self.proposal_generator.training = False
            unlabeled_prop, _ = self.proposal_generator(labeled_im, labeled_feat, None)
            self.proposal_generator.training = True

            ### Flip RPN proposals
            logger.debug("Flipping proposals")
            labeled_prop_flip = self._xflip_rpn_proposals(labeled_prop)
            unlabeled_prop_flip = self._xflip_rpn_proposals(unlabeled_prop)

        ### Standard supervised forward pass and loss accumulation for RoI heads
        # "supervised" argument below defines whether the supplied data has/needs GTs or not
        # and indicates whether to perform HNM; see :meth:`CSDStandardROIHeads.roi_heads`
        logger.debug("Performing a stadard forward pass")
        _, labeled_det_losses = self.roi_heads(labeled_im, labeled_feat, labeled_prop, labeled_gt, supervised=True)
        losses.update(labeled_det_losses)  # Save RoI heads supervised losses

        ### CSD forward pass and loss accumulation
        if calculate_csd:

            # Labeled inputs
            logger.debug("Performing a CSD forward pass for labeled inputs")
            labeled_csd_losses = self._csd_pass_and_get_loss(
                labeled_im,
                labeled_feat,
                labeled_prop,
                labeled_im_flip,
                labeled_feat_flip,
                labeled_prop_flip,
            )

            # Unlabeled inputs
            logger.debug("Performing a CSD forward pass for unlabeled inputs")
            unlabeled_csd_losses = self._csd_pass_and_get_loss(
                unlabeled_im,
                unlabeled_feat,
                unlabeled_prop,
                unlabeled_im_flip,
                unlabeled_feat_flip,
                unlabeled_prop_flip,
            )

            # Sum up the losses for labeled and unlabeled inputs and save to the loss dict
            for k in unlabeled_csd_losses:
                labeled_csd_losses[k] += unlabeled_csd_losses[k]

            losses.update(labeled_csd_losses)

        ### Original visualization code
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(labeled_inp, labeled_prop)

        logger.debug(f"Losses {losses}")

        return losses

    def _preprocess_images_and_get_gt(self, inputs):
        """D2's standard preprocessing of input instances.
        Moved to a separate method to avoid duplicate code."""

        images = self.preprocess_image(inputs)
        gt_instances = None
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in inputs]
        return images, gt_instances

    def _xflip_rpn_proposals(self, proposals: List[Instances]):
        """Creates a copy of given RPN proposals by flipping them along x-axis.

        Args:
            proposals: list of size N (where N is the number of images in the batch) of predicted
            instances from the RPN. Each element is a set of bboxes of type Instances.
        Returns:
            flipped_proposals: list of flipped instances in the original format.
        """

        # Create a new list for proposals
        proposals_new = []

        # See `d2.modeling.proposal_generator.proposal_utils.py` line 127 for the construction
        # of instances (to understand why we can apply this transform).
        # TODO: assert that RPN head is default one here or in config
        # Bboxes are in XYXY_ABS format from the default RPN head (see
        # `d2.modeling.anchor_generator.DefaultAnchorGenerator.generate_cell_anchors()`),
        # so we can apply the transformation right away. The exact content of proposals
        # (when using the default `RPN`) is defined on line 127 in `find_top_rpn_proposals`:
        # https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/proposal_generator/proposal_utils.py#L127
        for prop in proposals:
            im_size = prop._image_size  # Get image size

            prop_new = Instances(im_size)  # Create a new set of instances
            transform = T.HFlipTransform(im_size[1])  # Instantiate a transformation for the given image
            bbox_tensor = prop.proposal_boxes.tensor.detach().clone().cpu()  # Clone and detach bboxes
            bbox_tensor = transform.apply_box(bbox_tensor)  # Apply flip and send back to device
            bbox_tensor = torch.as_tensor(  # Convert back to Tensor on the correct device
                bbox_tensor, device=self.device
            )
            prop_new.proposal_boxes = Boxes(bbox_tensor)  # Save new bboxes
            # prop_new.objectness_logits = prop.objectness_logits.clone()

            proposals_new.append(prop_new)  # Save

        # TODO: check that flipping works

        return proposals_new

    def _csd_pass_and_get_loss(self, im, feat, prop, im_flip, feat_flip, prop_flip):
        """Passes images with RPN proposals and their flipped versions through RoI heads and calculates CSD loss.

        Args:
            im: list preprocessed images
            feat: list of backbone features for each image
            prop: list of Instances (object proposals from RPN) for each image
            im_flip, feat_flip, prop_flip: same data for flipped image
        Returns:
            dict: a dictionary with two keys containing CSD classification and localization losses.
        """

        ### Get raw RoI predictions on images and their flipped versions
        # Important note: because `prop_flip` is a modified element-wise copy of `prop, where for each proposal simply
        # coordinates were flipped but the order didn't change, we expect the model to return consistent scores for
        # each proposal, i.e. consistent class_scores and deltas; this is why in the code below we can simply do
        # e.g. `loss = mse(deltas - deltas_flip)`, as the matching is ensured
        ((class_scores, deltas), _) = self.roi_heads(im, feat, prop, supervised=False)
        ((class_scores_flip, deltas_flip), _) = self.roi_heads(im_flip, feat_flip, prop_flip, supervised=False)

        ### Calculate loss mask
        # Ignore bboxes for which background is the class with the largest probability
        # based on the non-flipped image
        bkg_scores = class_scores[:, -1]
        mask = bkg_scores < class_scores.max(1).values

        ### Calculate CSD classification loss
        csd_class_criterion = torch.nn.KLDivLoss().to(self.device)

        # Calculate KL losses between class_roi_head predictions for original and flipped versions
        # Note: the paper mentions JSD loss here, however, in the implementation authors mistakenly used
        # a sum of KL losses divided by two, which we reproduce here
        # TODO: check the shape of this
        csd_class_loss = (
            csd_class_criterion(class_scores[mask], class_scores_flip[mask]).sum(-1).mean()
            + csd_class_criterion(class_scores_flip[mask], class_scores[mask]).sum(-1).mean()
        ) / 2

        ### Calculate CSD localization losss
        # Note: default format of deltas is (dx, dy, dw, dh), see
        # :meth:`Box2BoxTransform.apply_deltas`.

        # Change the sign of predicted dx for flipped instances for simplified
        # loss calculation (see https://github.com/soo89/CSD-SSD/issues/3)
        deltas_flip[:, 0] = -deltas_flip[:, 0]

        # Calculate as MSE
        # TODO: check the shape of this
        csd_loc_loss = torch.mean(torch.pow(deltas[mask] - deltas_flip[mask], 2))

        return {
            "csd_loss_cls": csd_class_loss,
            "csd_loss_box_reg": csd_loc_loss,
        }
