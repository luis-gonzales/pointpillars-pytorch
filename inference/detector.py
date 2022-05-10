from typing import List

import numpy as np
from PIL import Image, ImageDraw
import torch

from anchors import generate_detections, Anchors
from data.pt_cloud import pt_cloud_to_pillars
from inference.inference_processing import post_process, box3d_to_bev
from model import PointPillars


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

        cls_out, box_out, indices, classes = post_process(
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
