from typing import List

import numpy as np
from PIL import Image, ImageDraw
import torch

from anchors import generate_detections, Anchors
from data.pt_cloud import pt_cloud_to_pillars
from model import PointPillars


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


def _post_process(
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


class PointPillarsInference:
    def __init__(self, ckpt_path, prob_threshold=0.3):
        model = PointPillars.load_from_checkpoint(checkpoint_path=ckpt_path)
        model.eval()

        hparams = model.hparams["hparams"]

        grid_range = (hparams.x_range[1] - hparams.x_range[0], hparams.y_range[1] - hparams.y_range[0])
        anchors = Anchors(min_level=0, max_level=0, num_scales=1,
            anchor_size=hparams.anchor_size, image_size=grid_range,
            z_center=hparams.anchor_zcenter, resolution=hparams.resolution*hparams.backbone_stride)

        self.model = model
        self.anchors = anchors
        self.hparams = hparams
        self.prob_threshold = prob_threshold

    def __call__(self, stacked_pillars, pillar_indices):
        stacked_pillars = torch.from_numpy(stacked_pillars).unsqueeze(dim=0)
        pillar_indices = torch.from_numpy(pillar_indices).unsqueeze(dim=0)

        box_out, cls_out = self.model(stacked_pillars, pillar_indices)

        cls_out, box_out, indices, classes = _post_process(
            [cls_out], [box_out], num_levels=1, num_classes=1)

        detections = generate_detections(
            cls_out[0], box_out[0], self.anchors.boxes, indices[0], classes[0], None, None)

        mask = detections[:, 7] > self.prob_threshold
        return detections[mask]


if __name__ == "__main__":
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument("ckpt", help="checkpoint file")
    # parser.add_argument("pt_cloud", help="path to dataset")
    args = parser.parse_args()

    model = PointPillarsInference(args.ckpt)
    config = model.hparams

    file = "sample_dataset/velodyne/000038.bin"
    # file = "sample_dataset/velodyne/001584.bin"
    # file = "sample_dataset/velodyne/002079.bin"
    # file = "sample_dataset/velodyne/003957.bin"
    pt_cloud = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
    stacked_pillars, pillar_indices = pt_cloud_to_pillars(pt_cloud, model.hparams)

    detections = model(stacked_pillars, pillar_indices)
    bev_dets = box3d_to_bev(detections[:, :7])

    img = Image.new("RGB", config.grid_shape[::-1], (255, 255, 255))
    drawer = ImageDraw.Draw(img)

    for bev_pts, conf, class_id in zip(bev_dets, detections[:, 7], detections[:, 8]):
        bev_pts[0] -= config.y_range[0]
        bev_pts /= config.resolution

        drawer.polygon(
            (tuple(bev_pts[:, 0]), tuple(bev_pts[:, 1]), tuple(bev_pts[:, 2]), tuple(bev_pts[:, 3])),
            outline=(0, 0, 0))

    img = img.rotate(180)

    img.save("inference.jpg")
