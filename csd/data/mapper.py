import copy

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data.dataset_mapper import DatasetMapper


class CSDDatasetMapper(DatasetMapper):
    """Yields augmented image and its flipped version.

    This customized mapper extends the default mapper (that applies ResizeShortestEdge and
    RandomFlip, see `detectron2.data.detection_utils.build_augmentation`) by additionally
    flipping the final image; it returns the image augmented in a default way along with its
    flipped version (for the CSD loss).
    The `__call__` method is a straightforward extension of the parent's one, most code is
    taken from there. See the `DatasetMapper` for more details.
    """

    def __call__(self, dataset_dict):
        """Loads image & attributes into the dict, returns a pair - for the original and the flipped ones.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            See full list of keys here: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

        Returns:
            tuple(dict, dict): a tuple where the first dict contains the data for the image augmented in a
            default way, and the second dict contains the same image but x-flipped

        Most of code comes from the original `__call__`. The only difference is the last few lines of code.
        There, the list of transforms is extended with an additional x-flip and its applied
        to the image. Note that it may happen that the resulting transforms list will have two x-flips
        (which is effectively no flip) and one may reason we could simply keep the original image untouched
        and flip its copy. However, we want to keep things as it is because only the original image (in the first
        dict) is used for the supervised training and the x-flipped image is used only for CSD loss. So if
        the original image would never get x-flipped, the model effectively will never be trained on x-flipped
        images.
        """

        # Load the image (D2's original code)
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        def apply_image_augmentations(image, dataset_dict, sem_seg_gt, augmentations):
            """Applies given augmentation to the given image and its attributes (segm, instances, etc).

            Almost no changes from D2's original code (apart from erasing non-relevant portions, e.g. for
            keypoints), just wrapped it in a function to avoid duplicate code."""

            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                for anno in dataset_dict["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                annos = [
                    utils.transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.instance_mask_format)

                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if self.recompute_boxes:
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = utils.filter_empty_instances(instances)

            return dataset_dict, transforms

        # Store the copies of image and its metadata for the future x-flip
        dataset_dict_flipped, image_flipped, sem_seg_gt_flipped = (
            dataset_dict.copy(),
            image.copy(),
            sem_seg_gt.copy() if sem_seg_gt else None,
        )

        # Augment the original image
        original_dataset_dict, transforms = apply_image_augmentations(
            image, dataset_dict, sem_seg_gt, self.augmentations
        )

        # Extend instantiated transforms with an additional x-flip in the end; see `TransformList.`__add__`
        transforms_w_flip = transforms + T.HFlipTransform(image.shape[1])
        # Transform Transforms to Augmentations; to learn more on how they differ you can check my note here:
        # https://www.notion.so/vlfom/How-augmentations-work-in-DatasetMapper-a4832df03489429ba04b9bc8d0e12dc6
        augs_w_flip = T.AugmentationList(transforms_w_flip)
        # Obtain the x-flipped data
        flipped_dataset_dict, _ = apply_image_augmentations(
            image_flipped, dataset_dict_flipped, sem_seg_gt_flipped, augs_w_flip
        )

        return (original_dataset_dict, flipped_dataset_dict)


class TestDatasetMapper(DatasetMapper):
    """A simple extension of `d2.data.DatasetMapper` that keeps annotations and segm_masks for test images.

    Default implementation removes all labels, however, they are needed for visualization purposes.
    The only difference from `d2.data.DatasetMapper.__call__` is that sosme lines are removed."""

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.instance_mask_format)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
