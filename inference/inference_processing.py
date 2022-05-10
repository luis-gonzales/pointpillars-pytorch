from typing import List

import numpy as np
import torch


def post_process(
        cls_outputs: List[torch.Tensor],
        box_outputs: List[torch.Tensor],
        num_levels: int,
        num_classes: int,
        max_detection_points: int = 5000,
):
    """Selects top-k predictions.
    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.
    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
        num_levels (int): number of feature levels
        num_classes (int): number of output classes
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, num_classes])
        for level in range(num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 7])
        for level in range(num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
    # FIXME change someday, will have to live with annoying warnings for a while as testing impl breaks torchscript
    # indices_all = torch.div(cls_topk_indices_all, num_classes, rounding_mode='trunc')
    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 7))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


def box3d_to_bev(detections):
    """
    convert x,y,z,w,l,h,theta,conf,class_id to bev coordinates
    """
    bev_detections = []
    for x, y, _, w, l, _, theta in detections:
        x, y, w, l, theta = [float(var) for var in [x, y, w, l, theta]]

        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [-s, -c]])   # 2x2

        pts = np.array([[-l/2, -l/2, l/2, l/2], [-w/2, w/2, w/2, -w/2]]) # 2x4
        pts = np.matmul(A, pts)
        pts += np.array([[y], [x]])         # offset
        bev_detections.append(pts)
    return np.array(bev_detections)
