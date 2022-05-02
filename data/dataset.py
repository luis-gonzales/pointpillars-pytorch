from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from anchors import Anchors, AnchorLabeler
from calibration import Calibration
from data.pt_cloud import pt_cloud_to_pillars
from data.transforms import train_transform


class PointPillarsCollate:
    def __init__(self, anchor_labeler, image_shape):
        self.anchor_labeler = anchor_labeler
        self.image_shape = image_shape

    def __call__(self, batch):
        """
        transform `batch` to tensors and encode with anchor_labeler
        """

        batch_size = len(batch)
        stacked_pillars = torch.zeros((batch_size, *batch[0]["stacked_pillars"].shape), dtype=torch.float32)
        pillar_indices = torch.zeros((batch_size, *batch[0]["pillar_indices"].shape), dtype=torch.int64)
        cls_targets = torch.zeros((batch_size, *self.image_shape, 2), dtype=torch.int64)    # 2 anchors (not ideal to be hardcoded)
        box_targets = torch.zeros((batch_size, *self.image_shape, 14), dtype=torch.float32) # 7 vars * 2 anchors (not ideal to be hardcoded)
        num_positives_targets = torch.zeros((batch_size), dtype=torch.float32)
        
        for i in range(batch_size):
            # input/point cloud
            stacked_pillars[i] = torch.from_numpy(batch[i]["stacked_pillars"])
            pillar_indices[i] = torch.from_numpy(batch[i]["pillar_indices"])

            # targets
            gt_boxes = batch[i]["gt_boxes"]
            gt_classes = batch[i]["gt_classes"]
            gt_boxes = torch.tensor(gt_boxes)
            gt_classes = torch.tensor(gt_classes)
            
            cls_targets_out, box_targets_out, num_positives = self.anchor_labeler.label_anchors(gt_boxes, gt_classes)
            cls_targets[i] = cls_targets_out[0]
            box_targets[i] = box_targets_out[0]
            num_positives_targets[i] = num_positives

        return {
            "stacked_pillars": stacked_pillars,
            "pillar_indices": pillar_indices,
            "cls_targets": cls_targets,
            "box_targets": box_targets,
            "num_positives": num_positives_targets}


class KittiDataset(Dataset):
    def __init__(self, root_dir, split, config, transform=None):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        with open(root_dir / "splits" / f"{split}.txt") as fp:
            self.files = fp.read().splitlines()

        self.root_dir = root_dir
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def _cam2_to_velo(self, coord, calib):
        coord_ref = calib.project_rect_to_ref(coord)
        return calib.project_ref_to_velo(coord_ref)

    def __getitem__(self, idx):
        D, P, N = 4, 12000, 100

        file_id = self.files[idx]
        calib = Calibration(self.root_dir / "calib" / f"{file_id}.txt")

        with open(self.root_dir / "label_2" / f"{file_id}.txt") as fp:
            anno = fp.read().splitlines()

        gt_boxes, gt_classes = [], []
        for obj in anno:
            obj_split = obj.split(" ")

            obj_type = obj_split[0]
            if obj_type != "Car":
                continue

            occluded = int(obj_split[2])    # 0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown
            h, w, l, x, y, z, rot_y = [float(obj_split[z]) for z in [8, 9, 10, 11, 12, 13, 14]]

            # convert from cam2 coordinates to velodyne/point cloud coordinates
            x_velo, y_velo, z_velo = self._cam2_to_velo(np.array([[x, y-h/2, z]]), calib)[0]

            gt_boxes.append([x_velo, y_velo, z_velo, l, w, h, -rot_y - np.pi/2])  # w is distance along x_velo, l is distance along y_velo
            gt_classes.append(1)

        if gt_boxes:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_classes = np.array(gt_classes, dtype=np.int64)
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_classes = np.array(gt_classes, dtype=np.int64)

        file = self.root_dir / "velodyne" / f"{file_id}.bin"
        pt_cloud = np.fromfile(file, dtype=np.float32).reshape((-1, 4))

        if self.transform:
            pt_cloud, gt_classes, gt_boxes = self.transform({
                "pt_cloud": pt_cloud,
                "classes": gt_classes,
                "boxes": gt_boxes})

        stacked_pillars, pillar_indices = pt_cloud_to_pillars(pt_cloud, self.config)

        return {
            "stacked_pillars": stacked_pillars,
            "pillar_indices": pillar_indices,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes}


class KittiDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, args):
        super().__init__()

        grid_range = (args.x_range[1] - args.x_range[0], args.y_range[1] - args.y_range[0])
        anchors = Anchors(min_level=0, max_level=0, num_scales=1, anchor_size=args.anchor_size,
            image_size=grid_range, z_center=args.anchor_zcenter, resolution=args.resolution)
        anchor_labeler = AnchorLabeler(anchors, pos_match_threshold=args.positive_threshold, neg_match_threshold=args.negative_threshold)
        grid_size = [int(z/args.resolution) for z in grid_range]

        self.collate_fn = PointPillarsCollate(anchor_labeler=anchor_labeler, image_shape=grid_size)
        self.batch_size = args.batch_size
        self.args = args
        self.root_dir = root_dir

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = KittiDataset(self.root_dir, "train", self.args)
            self.val_ds = KittiDataset(self.root_dir, "val", self.args)

    def train_dataloader(self):
        self.train_ds.transform = train_transform()
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, collate_fn=self.collate_fn, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, collate_fn=self.collate_fn, drop_last=False, pin_memory=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--x_range", nargs="+", default=[0.0, 70.4], type=float)    # space-separated in command-line
    parser.add_argument("--y_range", nargs="+", default=[-40.0, 40.0], type=float)  # space-separated in command-line
    parser.add_argument("--z_range", nargs="+", default=[-3.0, 1.0], type=float)  # space-separated in command-line
    parser.add_argument("--anchor_size", default=[1.6, 3.9, 1.5], type=float)    # w, l, h
    parser.add_argument("--anchor_zcenter", default=-1.0, type=float)
    parser.add_argument("--resolution", type=float, default=0.16)
    parser.add_argument("--positive_threshold", type=float, default=0.6)
    parser.add_argument("--negative_threshold", type=float, default=0.45)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dm = KittiDataModule("sample_dataset", args)
    dm.setup()
    train_dl = dm.train_dataloader()

    for x in train_dl:
        print("x:", type(x), x.keys())
        print('x["stacked_pillars"]:', type(x["stacked_pillars"]), x["stacked_pillars"].shape)

        cls_targets = x["cls_targets"]
        print("cls_targets:", cls_targets.shape)
        # print(torch.unique(cls_targets, return_counts=True))
        # print(cls_targets[0, 0, 0, :])

        cls_targets_0 = cls_targets[0]
        # print((cls_targets_0 >= 0).nonzero())

        cls_targets_1 = cls_targets[1]
        # print((cls_targets_1 >= 0).nonzero())
