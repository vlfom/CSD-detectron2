import copy
from typing import Any, Dict, List, Optional, Tuple

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from detectron2.config import global_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from PIL import Image


@META_ARCH_REGISTRY.register()
class CSDGeneralizedRCNN(GeneralizedRCNN):
    """Extends `GeneralizedRCNN`'s with additional logic for CSD."""

    def forward(
        self,
        batched_inputs_labeled: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        batched_inputs_unlabeled: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
        use_csd: Any = False,
    ):
        """Performs a standard forward pass along with CSD and returns resulting losses.

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

            batched_inputs_unlabeled (Optional): a list in a format similar to `batched_inputs_labeled`.
                Each item is based on one of the **unlabeled** images in the dataset.
                Therefore, the dict for each item contains only the `image` key without
                any ground truth instances.
                During the inference can be left unset or set to `None`.

            use_csd (Optional): a boolean that indicates whether to use CSD loss; it is `True` during all
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

        ### If in inference mode, return predictions for labeled inputs
        if not self.training:
            # By default evaluator passes just next(iter(dataloader)) to the model, where dataloader during evaluation
            # is the default one - it yields just a batch of one image (without labeled/unlabeled etc.
            # as done during training). Therefore, we can just use the first argument as an input to `inference`.
            # See `d2.evaluation.evaluator.inference_on_dataset` line 152.
            return self.inference(batched_inputs_labeled)

        losses = {}  # Placeholder for future loss accumulation

        # Indicates whether visualizations should be saved at this iteration
        do_visualize = comm.is_main_process() and self.vis_period and (get_event_storage().iter % self.vis_period == 0)

        ### Split labeled & unlabeled inputs and their flipped versions into separate variables
        labeled_inp, labeled_inp_flip = zip(*batched_inputs_labeled)
        unlabeled_inp, unlabeled_inp_flip = zip(*batched_inputs_unlabeled)

        ### Preprocess inputs
        # We need GTs only for labeled inputs, for others - ignore
        labeled_im, labeled_gt = self._preprocess_images_and_get_gt(labeled_inp)
        if use_csd:  # Don't need GTs here
            labeled_im_flip = self.preprocess_image(labeled_inp_flip)
            unlabeled_im = self.preprocess_image(unlabeled_inp)
            unlabeled_im_flip = self.preprocess_image(unlabeled_inp_flip)

        ### Backbone feature extraction
        # Extract features for all images and their flipped versions
        labeled_feat = self.backbone(labeled_im.tensor)
        if use_csd:
            labeled_feat_flip = self.backbone(labeled_im_flip.tensor)
            unlabeled_feat = self.backbone(unlabeled_im.tensor)
            unlabeled_feat_flip = self.backbone(unlabeled_im_flip.tensor)

        ### RPN proposals generation
        # NB: proposals are sorted by their objectness score in descending order
        # As described in the CSD paper, generate proposals only for non-flipped images
        labeled_prop, labeled_proposal_losses = self.proposal_generator(labeled_im, labeled_feat, labeled_gt)
        losses.update(labeled_proposal_losses)  # Save RPN losses

        if do_visualize:  # Visualize RPN proposals for labeled batch
            self._visualize_train_rpn_props(labeled_inp, labeled_prop)

        if use_csd:
            # For unlabeled images there is no GTs which would cause an error inside RPN;
            # however, we use a hack: set `training=False` temporarily and hope that it doesn't crash :)
            self.proposal_generator.training = False
            unlabeled_prop, _ = self.proposal_generator(unlabeled_im, unlabeled_feat, None)
            self.proposal_generator.training = True

            ### Flip RPN proposals
            labeled_prop_flip = self._xflip_rpn_proposals(labeled_prop)
            unlabeled_prop_flip = self._xflip_rpn_proposals(unlabeled_prop)

        ### Standard supervised forward pass and loss accumulation for RoI heads
        # "supervised" argument below defines whether the supplied data has/needs GTs or not
        # and indicates whether to perform HNM; see :meth:`CSDStandardROIHeads.roi_heads`

        if do_visualize:  # Visualize ROI predictions (**before** forward pass - it may change proposals)
            self._visualize_train_roi_preds(labeled_inp, labeled_im, labeled_feat, labeled_prop)

        _, labeled_det_losses = self.roi_heads(labeled_im, labeled_feat, labeled_prop, labeled_gt, supervised=True)
        losses.update(labeled_det_losses)  # Save RoI heads supervised losses

        ### CSD forward pass and loss accumulation
        if use_csd:
            # Labeled inputs
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

        ### Extra visualization step: predict RPN proposals and ROI bboxes for a fixed set of images
        if do_visualize:
            self._visualize_train_fixed_predictions(labeled_inp, labeled_im)

        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """Run inference on the given inputs (usually the batch size is 1 for inference).

        The code is taken from :meth:`GeneralizedRCNN.forward`, only some unnecessary if-else blocks were
        removed (not relevant for this project) and several lines to plot visualizations added,
        look for "CSD: ..." comments.
        """
        assert not self.training

        # Indicates whether visualizations should be generated for the current image
        do_visualize = comm.is_main_process() and self._inference_decide_visualization()

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, None)
        if do_visualize:
            self._visualize_test_rpn_props(batched_inputs, proposals)  # CSD: visualize RPN proposals

        results, _ = self.roi_heads(images, features, proposals, None)
        if do_visualize:
            self._visualize_test_roi_preds(batched_inputs, results)  # CSD: visualize ROI predictions

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def _inference_decide_visualization(self):
        """Helper method that decides whether an image should be visualized in inference mode.

        The idea is to visualize a random subset of images (`cfg.VIS_IMS_PER_GROUP` in total),
        however, because inside this object we can't access all the data, all we can do is use
        a "hacky" way - flip a coin for each image, and if we haven't reached the maximum
        allowed images - visualize this one.
        """

        if not global_cfg.VIS_TEST:  # Visualization is disabled in config
            return False

        if not hasattr(self, "_vis_test_counter"):  # Lazy initialization of "images already vis-ed" counter
            self._vis_test_counter = 0

        if self._vis_test_counter >= global_cfg.VIS_IMS_PER_GROUP:  # Enough images were plotted
            return False

        # Consider visualizing each ~100th image; this heuristic would work well for datasets where
        # #images is >> `100 * cfg.VIS_IMS_PER_GROUP`
        _r = np.random.randint(100)
        if _r == 0:
            self._vis_test_counter += 1
            return True
        return False

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
                proposals_new.append(prop_new)  # Save

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

        ### Calculate loss mask
        # Ignore bboxes for which background is the class with the largest probability
        # based on the non-flipped image
        bkg_scores = class_scores[:, -1]
        if len(class_scores) > 0:
            mask = bkg_scores < class_scores.max(1).values

        if (  # Predictions are empty or all bboxes are classified as bkg - return 0s
            len(class_scores) == 0 or mask.sum() == 0
        ):
            csd_class_loss = csd_loc_loss = torch.zeros(1).to(self.device)
        else:
            ### Calculate CSD classification loss
            csd_class_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True).to(self.device)

            # Calculate KL losses between class_roi_head predictions for original and flipped versions
            # Note: the paper mentions JSD loss here, however, in the implementation authors mistakenly used
            # a sum of KL losses divided by two, which we reproduce here
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
            csd_loc_loss = torch.mean(torch.pow(deltas[mask] - deltas_flip[mask], 2))

        return {
            f"{loss_dict_prefix}csd_loss_cls": csd_class_loss,
            f"{loss_dict_prefix}csd_loss_box_reg": csd_loc_loss,
        }

    def _visualize_train_rpn_props(self, inputs, props):
        """Visualizes region proposals from RPN during training in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            props,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="RPN",
        )

    def _visualize_train_roi_preds(self, inputs, ims, feats, props):
        """Visualizes predictions from ROI head during training in Wandb. See `_visualize_predictions` for more details."""

        # First, generate bboxes predictions; for that tell ROI head that we are in the inference mode
        # so that it yield instances instead of losses, etc.
        self.roi_heads.training = False
        with torch.no_grad():  # Make sure no gradients are changed
            pred_instances, _ = self.roi_heads(ims, feats, props, None, supervised=True)
        self.roi_heads.training = True

        self._visualize_predictions(
            inputs,
            pred_instances,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="ROI",
        )

    def _visualize_test_rpn_props(self, inputs, props):
        """Visualizes region proposals from RPN during inference in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            props,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="RPN_test",
            im_suffix=self._vis_test_counter,
        )

    def _visualize_test_roi_preds(self, inputs, pred_instances):
        """Visualizes predictions from ROI head during inference in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            pred_instances,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="ROI_test",
            im_suffix=self._vis_test_counter,
        )

    def _visualize_train_fixed_predictions(self, labeled_inp, labeled_im):
        """Visualize RPN proposals and ROI bboxes for a fixed set of images.

        The idea is to be able to monitor the progress on the same images, this helps to understand
        network's learning better.
        """

        # Run forward passes to generate proposals and bboxes without grads
        with torch.no_grad():
            # Check if such fixed set of images was initialized before; if not - create it
            if not hasattr(self, "_vis_imset"):
                self._vis_imset = (  # Not the optimal way to copy
                    copy.deepcopy(labeled_inp[: global_cfg.VIS_IMS_PER_GROUP]),
                    ImageList(
                        labeled_im.tensor[: global_cfg.VIS_IMS_PER_GROUP],
                        labeled_im.image_sizes[: global_cfg.VIS_IMS_PER_GROUP],
                    ),
                )

            feat = self.backbone(self._vis_imset[1].tensor)  # Extract backbone features

            self.proposal_generator.training = False  # Extract RPN proposals
            prop, _ = self.proposal_generator(self._vis_imset[1], feat, None)
            self.proposal_generator.training = True

            # Visualize RPN proposals
            self._visualize_predictions(
                self._vis_imset[0],
                prop,
                viz_count=global_cfg.VIS_IMS_PER_GROUP,
                max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
                predictions_mode="RPN_fixed",
            )

            self.roi_heads.training = False
            pred, _ = self.roi_heads(self._vis_imset[1], feat, prop, None, supervised=True)  # Get ROI predictions
            self.roi_heads.training = True

            # Visualize ROI predictions
            self._visualize_predictions(
                self._vis_imset[0],
                pred,
                viz_count=global_cfg.VIS_IMS_PER_GROUP,
                max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
                predictions_mode="ROI_fixed",
            )

    def _visualize_predictions(
        self, batched_inputs, predictions, predictions_mode, viz_count, max_predictions, im_suffix=None
    ):
        """Visualizes images and predictions and sends them to Wandb.

        batched_inputs: List[dict], list of inputs, each must contain keys "image"
            and "instances"
        predictions: List[Instances], list of bbox instances either from RPN or ROI head,
            see `_log_visualization_to_wandb` for more details.
        predictions_mode: str, must be begin with either 'RPN' or 'ROI', defines whether
            `predictions` contain bboxes predicted by RPN or ROI heads.
        """

        assert predictions_mode[:3] in ["RPN", "ROI"], "Unsupported proposal visualization mode"

        for inp, pred in zip(batched_inputs, predictions):
            img = inp["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            gts = inp["instances"]
            suffix = f"_{viz_count}" if im_suffix is None else im_suffix
            self._log_visualization_to_wandb(img, gts, pred, predictions_mode, max_predictions, im_suffix=suffix)

            viz_count -= 1  # Visualize up to `viz_count` images only
            if viz_count == 0:
                break

    def _log_visualization_to_wandb(self, image, gts, predictions, predictions_mode, max_predictions, im_suffix=""):
        """Logs provided image along with its GT and predicted bboxes to Wandb.

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
        if predictions_mode.startswith("ROI"):  # For ROI viz provide full vocabulary
            class_id_to_label = MetadataCatalog.get(global_cfg.DATASETS.TRAIN[0]).thing_classes[:]
            class_id_to_label = {k: v for k, v in enumerate(class_id_to_label)}  # Convert to dict (Wandb req)
        elif predictions_mode.startswith("RPN"):  # For RPN viz provide only "meta-labels"
            class_id_to_label = {1: "object", 2: "proposal"}

        viz_meta = {  # To store the vizualization metadata
            "predictions": {"box_data": [], "class_labels": class_id_to_label},
            "ground_truth": {"box_data": [], "class_labels": class_id_to_label},
        }

        # Append GT bboxes
        gt_bboxes = gts.gt_boxes.tensor
        gt_classes = gts.gt_classes
        for i in range(len(gts)):
            viz_meta["ground_truth"]["box_data"].append(
                self._bbox_to_wandb_dict(
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
                    self._bbox_to_wandb_dict(
                        bboxes[i],
                        2,  # RPN's proposal ID
                        class_id_to_label,
                        scores={"prob": float(probs[i])},
                        image_shape=image.shape,
                    )
                )
        elif predictions_mode.startswith("ROI"):
            # Limit the number of proposals (sorted by objectness score)
            box_number = min(len(predictions.pred_boxes), max_predictions)
            bboxes = predictions.pred_boxes.tensor[:box_number]
            probs = predictions.scores
            classes = predictions.pred_classes
            for i in range(len(bboxes)):
                viz_meta["predictions"]["box_data"].append(
                    self._bbox_to_wandb_dict(
                        bboxes[i],
                        int(classes[i]),
                        class_id_to_label,
                        scores={"prob": float(probs[i])},
                        image_shape=image.shape,
                    )
                )

        try:  # Get current iteration
            iter_ = get_event_storage().iter
        except:  # There is no iter when in eval mode - set to
            iter_ = 0

        wandb_img = wandb.Image(image, boxes=viz_meta)  # Send to wandb
        wandb.log({f"{predictions_mode}_predictions{im_suffix}": wandb_img, "global_step": iter_}, step=iter_)

    def _bbox_to_wandb_dict(self, xyxy_bbox, class_id, class_id_to_label, image_shape, scores=None):
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
