""" RetinaNet / EfficientDet Anchor Gen

*** adapted from https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/anchors.py ***

Adapted for PyTorch from Tensorflow impl at
    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py

Hacked together by Ross Wightman, original copyright below
"""
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Anchor definition.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms

from object_detection.argmax_matcher import ArgMaxMatcher
from object_detection.box_coder import FasterRcnnBoxCoder
from object_detection.box_list import BoxList
from object_detection.region_similarity_calculator import IouSimilarity
from object_detection.target_assigner import TargetAssigner


def decode_box_outputs(rel_codes, anchors):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.
    """
    xcenter_a, ycenter_a, zcenter_a, wa, la, ha = anchors.unbind(dim=1)

    tx, ty, tz, tw, tl, th = rel_codes.unbind(dim=1)

    xcenter = tx * torch.sqrt(wa**2 + la**2) + xcenter_a
    ycenter = ty * torch.sqrt(wa**2 + la**2) + ycenter_a
    zcenter = tz * ha + zcenter_a
    w = torch.exp(tw) * wa
    l = torch.exp(tl) * la
    h = torch.exp(th) * ha

    return torch.stack([xcenter, ycenter, zcenter, w, l, h], dim=1)


def generate_detections(
        cls_outputs, box_outputs, anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor], img_size: Optional[torch.Tensor],
        max_det_per_image: int = 100, soft_nms: bool = False):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels.

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [max_det_per_image, 6],
            each row representing [x_min, y_min, x_max, y_max, score, class]
    """
    assert box_outputs.shape[-1] == 6
    assert anchor_boxes.shape[-1] == 6
    assert cls_outputs.shape[-1] == 1

    anchor_boxes = anchor_boxes[indices, :]

    # Apply bounding box regression to anchors, boxes are converted to xyxy
    # here since PyTorch NMS expects them in that form.
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes)
    if img_scale is not None and img_size is not None:
        boxes = clip_boxes_xyxy(boxes, img_size / img_scale)  # clip before NMS better?

    scores = cls_outputs.sigmoid().squeeze(1).float()

    # batched_nms expects Nx4 of (x1, y1, x2, y2); boxes are (x, y, z, w, l, h)
    nms_boxes = boxes[:, [0, 1, 0, 1]]
    nms_boxes[:, [0, 1]] -= boxes[:, [3, 4]] / 2
    nms_boxes[:, [2, 3]] += boxes[:, [3, 4]] / 2

    if soft_nms:
        top_detection_idx, soft_scores = batched_soft_nms(
            boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=.001)
        scores[top_detection_idx] = soft_scores
    else:
        top_detection_idx = batched_nms(nms_boxes, scores, classes, iou_threshold=0.5)

    # keep only top max_det_per_image scoring predictions
    top_detection_idx = top_detection_idx[:max_det_per_image]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx, None]
    classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0

    if img_scale is not None:
        boxes = boxes * img_scale

    # FIXME add option to convert boxes back to yxyx? Otherwise must be handled downstream if
    # that is the preferred output format.

    # stack em and pad out to max_det_per_image if necessary
    num_det = len(top_detection_idx)
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if num_det < max_det_per_image:
        detections = torch.cat([
            detections,
            torch.zeros((max_det_per_image - num_det, 6), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections   # x, y, z, w, l, h, conf, class_idx


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, anchor_size, image_size: Tuple[int, int], z_center, resolution):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: Sequence specifying input image size of model (H, W).
                The image_size should be divided by the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        # self.aspect_ratios = aspect_ratios
        self.z_center = z_center
        self.resolution = resolution
        self.anchor_size = anchor_size

        assert isinstance(image_size, Sequence) and len(image_size) == 2
        image_size = [int(x/resolution) for x in image_size]
        self.image_size = tuple(image_size)
        self.feat_sizes = get_feat_sizes(image_size, max_level)
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    @classmethod
    def from_config(cls, config):
        return cls(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        anchor_sizes = [self.anchor_size, [self.anchor_size[z] for z in [1, 0, 2]]]
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for anchor_size in anchor_sizes:
                anchor_configs[level].append((anchor_size))
        return anchor_configs

    def _generate_boxes(self):
        """Generates multi-scale anchor boxes."""
        res = self.resolution

        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = []
            for anchor_size_x, anchor_size_y, anchor_size_z in configs:
                y = np.arange(res / 2, self.image_size[0] * res, res)
                x = np.arange(-self.image_size[1] * res / 2 + res / 2, self.image_size[1] * res / 2, res)

                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                zv = np.repeat(self.z_center, xv.shape)

                anchor_size_x = np.repeat(anchor_size_x, xv.shape)
                anchor_size_y = np.repeat(anchor_size_y, xv.shape)
                anchor_size_z = np.repeat(anchor_size_z, xv.shape)

                boxes = np.vstack((yv, xv, zv, anchor_size_x, anchor_size_y, anchor_size_z))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 6]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).float()
        return anchor_boxes

    def get_anchors_per_location(self):
        # return self.num_scales * len(self.aspect_ratios)
        return self.num_scales * 2  # 0, 90 degrees (not ideal to be hardcoded)


class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, pos_match_threshold, neg_match_threshold):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        similarity_calc = IouSimilarity()
        matcher = ArgMaxMatcher(
            pos_match_threshold,
            unmatched_threshold=neg_match_threshold,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=False)
        box_coder = FasterRcnnBoxCoder()

        self.target_assigner = TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.indices_cache = {}

    def label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_classes: A integer tensor with shape [N, 1] representing groundtruth classes.

            filter_valid: Filter out any boxes w/ gt class <= -1 before assigning

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        cls_targets_out = []
        box_targets_out = []

        if filter_valid:
            valid_idx = gt_classes > -1  # filter gt targets w/ label <= -1
            gt_boxes = gt_boxes[valid_idx]
            gt_classes = gt_classes[valid_idx]

        # ignore height and elevation for matching
        cls_targets, box_targets, matches = self.target_assigner.assign(
            BoxList(self.anchors.boxes), BoxList(gt_boxes), gt_classes)

        # class labels start from 1 and the background class = -1
        cls_targets = (cls_targets - 1).long()

        # Unpack labels.
        """Unpacks an array of cls/box into multiple scales."""
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.anchors.feat_sizes[level]
            steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
            cls_targets_out.append(cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            box_targets_out.append(box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            count += steps

        num_positives = (matches.match_results > -1).float().sum()

        return cls_targets_out, box_targets_out, num_positives


if __name__ == "__main__":
    from inference import _post_process


    w, h = 80.0, 70.4   # car x, y, z range is [(0, 70.4), (-40, 40), (-3, 1)]

    anchors = Anchors(
        min_level=0,
        max_level=0,
        num_scales=1,
        anchor_size=(1.6, 3.9, 1.5), # w, l, h
        image_size=(h, w),
        z_center=-1,
        resolution=0.16)

    print("anchors.boxes:")
    print(anchors.boxes)    # x, y, z, w, l, h (w, l is birdseye view)

    gt_boxes = [[58.49, -16.53, 2.39, 1.87, 3.69, 1.67]]    # x, y, z, w, l, h
    gt_classes = [1]

    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    gt_classes = np.array(gt_classes, dtype=np.int64)

    gt_boxes = torch.tensor(gt_boxes)
    gt_classes = torch.tensor(gt_classes)

    anchor_labeler = AnchorLabeler(anchors, pos_match_threshold=0.6, neg_match_threshold=0.45)

    cls_targets_out, box_targets_out, num_positives = anchor_labeler.label_anchors(gt_boxes, gt_classes)
    print("cls_targets_out:", type(cls_targets_out), [z.shape for z in cls_targets_out])
    print("box_targets_out:", type(box_targets_out), [z.shape for z in box_targets_out])
    print("num_positives =", num_positives)        

    print("cls_targets_out:", torch.unique(cls_targets_out[0], return_counts=True))

    print("indices of positive class:")
    print((cls_targets_out[0] >= 0).nonzero())
    print("cls_targets_out:", cls_targets_out[0].shape)
    
    cls_targets_out = [torch.unsqueeze(x, dim=0) for x in cls_targets_out]
    box_targets_out = [torch.unsqueeze(x, dim=0) for x in box_targets_out]

    # need to permute to inference output
    cls_out, box_out, indices, classes = _post_process(
        cls_targets_out, box_targets_out, num_levels=1, num_classes=1)

    print("detection output:")
    detections = generate_detections(
        cls_out[0], box_out[0], anchors.boxes, indices[0], classes[0], None, None)

    print("detections:")
    print(detections)
