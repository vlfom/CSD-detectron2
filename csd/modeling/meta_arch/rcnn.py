from typing import Any, Dict, List, Tuple

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.nn.functional as F
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
        use_csd: Any = False,
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

            use_csd: a boolean that indicates whether to use CSD loss; it is `True` during all
                training iterations apart from few initial ones, when CSD loss is not calculated,
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

        self.logger = setup_logger(name=__name__)  # TODO: remove all logging from here
        self.log = False

        if self.log:
            storage = get_event_storage()
            self.logger.debug(f"RCNN forward pass for iteration {storage.iter}")
            torch.set_printoptions(linewidth=190, edgeitems=10)

        losses = {}  # Placeholder for future loss accumulation

        ### Split labeled & unlabeled inputs and their flipped versions into separate variables
        if self.log:
            self.logger.debug("Split inputs")
        labeled_inp, labeled_inp_flip = zip(*batched_inputs_labeled)
        unlabeled_inp, unlabeled_inp_flip = zip(*batched_inputs_unlabeled)
        if self.log:
            self.logger.debug(
                "Size of labeled and unlabeled data: {} {}, {} {}".format(
                    len(labeled_inp), len(labeled_inp_flip), len(unlabeled_inp), len(unlabeled_inp_flip)
                )
            )

        ### If in inference mode, return predictions for labeled inputs (skipping unlabeled batch)
        if not self.training:
            return self.inference(labeled_inp)

        ### Preprocess inputs
        # We need GTs only for labeled inputs, for others - ignore
        if self.log:
            self.logger.debug("Preprocess inputs")
        labeled_im, labeled_gt = self._preprocess_images_and_get_gt(labeled_inp)
        if use_csd:
            labeled_im_flip, _ = self._preprocess_images_and_get_gt(labeled_inp_flip)
            if self.log:
                self.logger.debug(
                    "Labeled image sizes: {}, image content shape: {}".format(
                        labeled_im.image_sizes[0:2], labeled_im.tensor.shape
                    )
                )
                self.logger.debug(
                    "Labeled-flip image sizes: {}, image content shape: {}".format(
                        labeled_im_flip.image_sizes[0:2], labeled_im_flip.tensor.shape
                    )
                )
                self.logger.debug(
                    "Labeled image content:\n{}".format(
                        labeled_im.tensor[0, 0, :1, : (labeled_im.image_sizes[0][1] + 1)]
                    )
                )
                self.logger.debug(
                    "Labeled-flip image content:\n{}".format(
                        labeled_im_flip.tensor[0, 0, :1, : (labeled_im_flip.image_sizes[0][1] + 1)]
                    )
                )
            # For unlabeled inputs, no GTs exist - ignore
            unlabeled_im, _ = self._preprocess_images_and_get_gt(unlabeled_inp)
            unlabeled_im_flip, _ = self._preprocess_images_and_get_gt(unlabeled_inp_flip)
            if self.log:
                self.logger.debug(
                    "Unlabeled image content:\n{}".format(
                        unlabeled_im.tensor[0, 0, :1, : (unlabeled_im.image_sizes[0][1] + 1)]
                    )
                )
                self.logger.debug(
                    "Unlabeled-flip image content:\n{}".format(
                        unlabeled_im_flip.tensor[0, 0, :1, : (unlabeled_im_flip.image_sizes[0][1] + 1)]
                    )
                )

        ### Backbone feature extraction
        # Extract features for all images and their flipped versions
        if self.log:
            self.logger.debug("Backbone feature extraction")
        labeled_feat = self.backbone(labeled_im.tensor)
        if use_csd:
            labeled_feat_flip = self.backbone(labeled_im_flip.tensor)
            unlabeled_feat = self.backbone(unlabeled_im.tensor)
            unlabeled_feat_flip = self.backbone(unlabeled_im_flip.tensor)

            if self.log:
                f_key = list(labeled_feat.keys())[0]
                self.logger.debug(
                    "Labeled backbone content: {}\n{}".format(
                        labeled_feat[f_key].shape,
                        labeled_feat[f_key][0, 0, :1, : labeled_im.image_sizes[0][1] // 4 + 5],
                    )
                )
                self.logger.debug(
                    "Labeled-flip backbone content: {}\n{}".format(
                        labeled_feat_flip[f_key].shape,
                        labeled_feat_flip[f_key][0, 0, :1, : labeled_im.image_sizes[0][1] // 4 + 5],
                    )
                )

        ### RPN proposals generation
        if self.log:
            self.logger.debug("Generating proposals for labeled")
        # As described in the CSD paper, generate proposals only for non-flipped images
        labeled_prop, labeled_proposal_losses = self.proposal_generator(labeled_im, labeled_feat, labeled_gt)
        losses.update(labeled_proposal_losses)  # Save RPN losses

        if use_csd:
            # For unlabeled images there is no GTs which would cause an error inside RPN;
            # however, we use a hack: set `training=False` temporarily and hope that it doesn't crash :)
            # TODO: check that it works
            if self.log:
                self.logger.debug("Generating proposals for unlabeled")
            self.proposal_generator.training = False
            unlabeled_prop, _ = self.proposal_generator(unlabeled_im, unlabeled_feat, None)
            self.proposal_generator.training = True

            if self.log:
                self.logger.debug("Labeled proposals:")
                self.logger.debug(
                    "#batch {}\tprops for #0 {}\treq_grad {}".format(
                        len(labeled_prop),
                        list(labeled_prop[0].proposal_boxes.tensor.shape),
                        labeled_prop[0].proposal_boxes.tensor.requires_grad,
                    )
                )
                self.logger.debug("proposal #0 {}".format(labeled_prop[0].proposal_boxes.tensor[0]))

            ### Flip RPN proposals
            if self.log:
                self.logger.debug("Flipping proposals for both labeled and unlabeled")
            labeled_prop_flip = self._xflip_rpn_proposals(labeled_prop)
            unlabeled_prop_flip = self._xflip_rpn_proposals(unlabeled_prop)
            if self.log:
                self.logger.debug("Labeled-flip proposals:")
                self.logger.debug(
                    "#batch {}\tprops for #0 {}\treq_grad {}".format(
                        len(labeled_prop_flip),
                        list(labeled_prop_flip[0].proposal_boxes.tensor.shape),
                        labeled_prop_flip[0].proposal_boxes.tensor.requires_grad,
                    )
                )
                self.logger.debug("proposal #0 {}".format(labeled_prop_flip[0].proposal_boxes.tensor[0]))

        ### Standard supervised forward pass and loss accumulation for RoI heads
        # "supervised" argument below defines whether the supplied data has/needs GTs or not
        # and indicates whether to perform HNM; see :meth:`CSDStandardROIHeads.roi_heads`
        if self.log:
            self.logger.debug("Performing a standard forward pass of RoIs")
        _, labeled_det_losses = self.roi_heads(labeled_im, labeled_feat, labeled_prop, labeled_gt, supervised=True)
        losses.update(labeled_det_losses)  # Save RoI heads supervised losses

        ### CSD forward pass and loss accumulation
        if use_csd:
            # Labeled inputs
            if self.log:
                self.logger.debug("Performing a CSD forward pass for labeled inputs")
            labeled_csd_losses = self._csd_pass_and_get_loss(
                labeled_im,
                labeled_feat,
                labeled_prop,
                labeled_im_flip,
                labeled_feat_flip,
                labeled_prop_flip,
                loss_dict_prefix="sup_",
            )
            losses.update(labeled_csd_losses)  # Update the loss dict

            # Unlabeled inputs
            if self.log:
                self.logger.debug("Performing a CSD forward pass for unlabeled inputs")
            unlabeled_csd_losses = self._csd_pass_and_get_loss(
                unlabeled_im,
                unlabeled_feat,
                unlabeled_prop,
                unlabeled_im_flip,
                unlabeled_feat_flip,
                unlabeled_prop_flip,
                loss_dict_prefix="unsup_",
            )
            losses.update(unlabeled_csd_losses)  # Update the loss dict
        else:  # Set CSD losses to zeros when CSD is not used
            tzero = torch.zeros(1).to(self.device)
            for k in ["sup_csd_loss_cls", "sup_csd_loss_box_reg", "unsup_csd_loss_cls", "unsup_csd_loss_box_reg"]:
                losses[k] = tzero

        ### Original visualization code
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(labeled_inp, labeled_prop)

        if self.log:
            self.logger.debug(f"Losses {losses}")

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

        if self.log:
            self.logger.debug("[_xflip_rpn_proposals] in length: {}".format(len(proposals)))

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
        with torch.no_grad():
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

        if self.log:
            self.logger.debug("[_xflip_rpn_proposals] out length: {}".format(len(proposals_new)))

        return proposals_new

    def _csd_pass_and_get_loss(self, im, feat, prop, im_flip, feat_flip, prop_flip, loss_dict_prefix):
        """Passes images with RPN proposals and their flipped versions through RoI heads and calculates CSD loss.

        Args:
            im: list preprocessed images
            feat: list of backbone features for each image
            prop: list of Instances (object proposals from RPN) for each image
            im_flip, feat_flip, prop_flip: same data for flipped image
            loss_dict_prefix: prefix for the loss dict to store the losses
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

        ### Apply log-softmax to class-probabilities
        class_scores = F.log_softmax(class_scores, dim=1)
        class_scores_flip = F.log_softmax(class_scores_flip, dim=1)

        if self.log:
            self.logger.debug(
                "[_csd_pass_and_get_loss] class_scores original: {}\n{}".format(class_scores.shape, class_scores[:3])
            )
            self.logger.debug(
                "[_csd_pass_and_get_loss] class_scores flipped: {}\n{}".format(
                    class_scores_flip.shape, class_scores_flip[:3]
                )
            )

        ### Calculate loss mask
        # Ignore bboxes for which background is the class with the largest probability
        # based on the non-flipped image
        bkg_scores = class_scores[:, -1]
        mask = bkg_scores < class_scores.max(1).values

        if mask.sum() == 0:  # All bboxes are classified as bkg - return 0s
            csd_class_loss = csd_loc_loss = torch.zeros(1).to(self.device)
        else:
            if self.log:
                self.logger.debug(
                    "[_csd_pass_and_get_loss] class_scores original after mask: {}\n{}".format(
                        class_scores[mask].shape, class_scores[mask][:3]
                    )
                )
                self.logger.debug(
                    "[_csd_pass_and_get_loss] class_scores flipped after mask: {}\n{}".format(
                        class_scores_flip[mask].shape, class_scores_flip[mask][:3]
                    )
                )

            ### Calculate CSD classification loss
            csd_class_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True).to(self.device)

            if self.log:
                self.logger.debug(
                    "[_csd_pass_and_get_loss] class_loss one-sided:\n{}".format(
                        csd_class_criterion(class_scores[mask], class_scores_flip[mask])
                    )
                )

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

            if self.log:
                self.logger.debug(
                    "[_csd_pass_and_get_loss] class_reg one-sided:\n{}".format(
                        csd_class_criterion(class_scores[mask], class_scores_flip[mask])
                    )
                )

        return {
            f"{loss_dict_prefix}csd_loss_cls": csd_class_loss,
            f"{loss_dict_prefix}csd_loss_box_reg": csd_loc_loss,
        }
