from argparse import ArgumentParser
import json
from pathlib import Path
import tempfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.dataset import KittiDataset
from inference.detector import PointPillarsInference
from inference.inference_processing import box3d_to_bev


parser = ArgumentParser()
parser.add_argument("ckpt", help="checkpoint file")
parser.add_argument("ds_path", help="path to dataset")
args = parser.parse_args()

model = PointPillarsInference(args.ckpt, prob_threshold=0.1)
config = model.hparams

val_ds = KittiDataset(args.ds_path, "val", config)

predictions = []
annotations, images = [], []
obj_id = 1
for sample in val_ds:
    images.append({"id": sample["file_id"]})

    # ground-truth
    gt_boxes = sample["gt_boxes"]
    gt_classes = sample["gt_classes"]
    bev_anns = box3d_to_bev(gt_boxes)
    for gt_cls, bev_pts in zip(gt_classes, bev_anns):
        right, left = np.min(bev_pts[0]), np.max(bev_pts[0])
        bottom, top = np.min(bev_pts[1]), np.max(bev_pts[1])
        w, l = left - right, top - bottom

        annotations.append({
            "image_id": sample["file_id"],
            "id": obj_id,
            "bbox": [left, top, w, l],
            "category_id": int(gt_cls),
            "iscrowd": False,       # required by pycocotools
            "area": w*l})           # required by pycocotools but doesn't seem to impact results
        obj_id += 1

    # detections
    detections = model(sample["stacked_pillars"], sample["pillar_indices"])
    bev_dets = box3d_to_bev(detections[:, :7])
    for bev_pts, conf, class_id in zip(bev_dets, detections[:, 7], detections[:, 8]):
        right, left = np.min(bev_pts[0]), np.max(bev_pts[0])
        bottom, top = np.min(bev_pts[1]), np.max(bev_pts[1])
        predictions.append({
            "image_id": sample["file_id"],
            "bbox": [left, top, left-right, top-bottom],
            "score": float(conf),
            "category_id": int(class_id)})

with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp_gt:
    json.dump({
        "annotations": annotations,
        "images": images,
        "categories": [{"name": "Car", "id": 1}]}, fp_gt)

with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp_dt:
    json.dump(predictions, fp_dt)

coco_gt = COCO(fp_gt.name)
coco_det = coco_gt.loadRes(fp_dt.name)
coco_eval = COCOeval(coco_gt, coco_det, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# close tmp files
Path.unlink(Path(fp_dt.name))
Path.unlink(Path(fp_gt.name))
