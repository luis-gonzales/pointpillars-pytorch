import math

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import nn
import wandb

from loss import DetectionLoss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def scatter(pillars, indices, config):
    pillars = pillars.permute(0, 2, 1)          # (batch,p,c)

    # roll into single dimension for later _index_add
    indices[:, :, 0] *= config.grid_shape[1]    # width
    indices = torch.sum(indices, dim=2)         # (batch,p)

    device = pillars.device

    # process per batch!
    batch_size = indices.shape[0]
    output = torch.zeros((batch_size, config.grid_shape[0]*config.grid_shape[1], config.encoder_channels)).to(device)
    for k in range(batch_size):
        output[k] = torch.zeros((config.grid_shape[0]*config.grid_shape[1], config.encoder_channels), dtype=pillars.dtype).to(device).index_add_(0, indices[k], pillars[k])

    # reshape to pseudo-image dims
    output = output.reshape((-1, config.grid_shape[0], config.grid_shape[1], config.encoder_channels))  # b,h,w,c
    output = output.permute(0, 3, 1, 2)

    return output


class PillarFeatureNet(nn.Module):
    def __init__(self, config):
        super(PillarFeatureNet, self).__init__()

        D, N = 4, 100

        self.linear = nn.Conv2d(D, config.encoder_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(config.encoder_channels)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d((1, N))

        self.config = config

    def forward(self, x, indices):
        # linear + bn + relu: D,P,N > C,P,N
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        # max op over channels: C,P,N > C,P
        x = self.max_pool(x)
        x = x.squeeze(-1)

        # scatter: batch,C,P > batch,C,H,W
        x = scatter(x, indices, self.config)

        return x


class Block(nn.Module):
    def __init__(self, s_in, s, l, f_in, f):
        super(Block, self).__init__()

        padding = 1
        self.convs = nn.ModuleList()
        for k in range(l):
            stride = int(s/s_in) if k == 0 else 1
            self.convs.append(nn.Conv2d(f_in, f, 3, stride=stride, padding=padding, bias=False))
            f_in = f

        self.bns = nn.ModuleList([nn.BatchNorm2d(f) for _ in range(l)])

        self.relu = nn.ReLU()

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        return x


class Up(nn.Module):
    def __init__(self, s_in, s_out, f_in, f):
        super(Up, self).__init__()
        
        stride = int(s_in/s_out)
        self.conv_transpose = nn.ConvTranspose2d(f_in, f, kernel_size=3, stride=stride, padding=1, output_padding=stride-1)
        self.bn = nn.BatchNorm2d(f)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        encoder_channels = config.encoder_channels
        backbone_stride = config.backbone_stride

        s = [backbone_stride, 2*backbone_stride, 4*backbone_stride]
        l = [4, 6, 6]
        f = [encoder_channels, 2*encoder_channels, 4*encoder_channels]

        self.block1 = Block(1, s[0], l[0], encoder_channels, f[0])
        self.block2 = Block(s[0], s[1], l[1], f[0], f[1])
        self.block3 = Block(s[1], s[2], l[2], f[1], f[2])

        self.up1 = Up(backbone_stride, backbone_stride, f[0], f[1])
        self.up2 = Up(2*backbone_stride, backbone_stride, f[1], f[1])
        self.up3 = Up(4*backbone_stride, backbone_stride, f[2], f[1])

    def forward(self, x):
        x_block1 = self.block1(x)
        x_block2 = self.block2(x_block1)        
        x_block3 = self.block3(x_block2)

        x_up1 = self.up1(x_block1)
        x_up2 = self.up2(x_block2)
        x_up3 = self.up3(x_block3)

        return torch.cat((x_up1, x_up2, x_up3), dim=1)


class HeadNet(nn.Module):
    def __init__(self, out_channels, config):
        super(HeadNet, self).__init__()

        encoder_channels = config.encoder_channels
        self.pred = nn.Conv2d(6*encoder_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.pred(x)


def _init_weight(m, n=""):
    if isinstance(m, nn.Conv2d):
        if "cls_net" in n:
            m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))


class PointPillars(pl.LightningModule):
    def __init__(self, hparams):
        super(PointPillars, self).__init__()

        # grid size of pseudo-image based on x,y range (w, h)
        hparams.grid_shape = hparams.x_range[1] - hparams.x_range[0], hparams.y_range[1] - hparams.y_range[0]
        hparams.grid_shape = tuple([int(z/hparams.resolution) for z in hparams.grid_shape])

        self.save_hyperparameters()                     # access through self.hparams["hparams"]

        num_anchors = 2                                 # 0, 90 degrees
        self.pillar_feature_net = PillarFeatureNet(hparams)
        self.backbone = Backbone(hparams)
        self.box_net = HeadNet(num_anchors*7, hparams)  # x, y, z, w, l, h, theta
        self.cls_net = HeadNet(num_anchors*hparams.num_classes, hparams)

        for n, m in self.named_modules():
            _init_weight(m, n)

        self.loss = DetectionLoss(hparams)

        self.cls_loss_train_m = AverageMeter()
        self.box_loss_train_m = AverageMeter()
        self.cls_loss_val_m = AverageMeter()
        self.box_loss_val_m = AverageMeter()

    def forward(self, x, indices):
        x = self.pillar_feature_net(x, indices)
        x = self.backbone(x)
        cls_out = self.cls_net(x)
        box_out = self.box_net(x)
        return box_out, cls_out

    def training_step(self, batch, batch_idx):
        stacked_pillars = batch["stacked_pillars"]
        pillar_indices = batch["pillar_indices"]

        box_out, cls_out = self(stacked_pillars, pillar_indices)

        cls_targets = batch["cls_targets"]
        box_targets = batch["box_targets"]
        num_positives = batch["num_positives"]

        loss, cls_loss, box_loss = self.loss([cls_out], [box_out], [cls_targets], [box_targets], num_positives)

        self.box_loss_train_m.update(box_loss)
        self.cls_loss_train_m.update(cls_loss)

        return loss

    def on_train_epoch_start(self):
        self.box_loss_train_m.reset()
        self.cls_loss_train_m.reset()

    def on_train_epoch_end(self, *args, **kwargs):
        wandb.log({
            "cls_loss_train": self.cls_loss_train_m.avg,
            "box_loss_train": self.box_loss_train_m.avg,
            "cls_loss_val": self.cls_loss_val_m.avg,
            "box_loss_val": self.box_loss_val_m.avg})

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        stacked_pillars = batch["stacked_pillars"]
        pillar_indices = batch["pillar_indices"]

        box_out, cls_out = self(stacked_pillars, pillar_indices)

        cls_targets = batch["cls_targets"]
        box_targets = batch["box_targets"]
        num_positives = batch["num_positives"]

        loss, cls_loss, box_loss = self.loss([cls_out], [box_out], [cls_targets], [box_targets], num_positives)

        self.cls_loss_val_m.update(cls_loss)
        self.box_loss_val_m.update(box_loss)

    def on_val_epoch_start(self):
        self.cls_loss_val_m.reset()
        self.box_loss_val_m.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["hparams"].lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PointPillars")
        parser.add_argument("--resolution", type=float, default=0.16)
        parser.add_argument("--encoder_channels", type=int, default=64)
        parser.add_argument("--x_range", nargs="+", default=[0.0, 70.4], type=float)    # space-separated in command-line
        parser.add_argument("--y_range", nargs="+", default=[-40.0, 40.0], type=float)  # space-separated in command-line
        parser.add_argument("--z_range", nargs="+", default=[-3.0, 1.0], type=float)    # space-separated in command-line
        parser.add_argument("--backbone_stride", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--anchor_size", default=[1.6, 3.9, 1.5], type=float)       # w, l, h
        parser.add_argument("--anchor_zcenter", default=-1.0, type=float)
        parser.add_argument("--positive_threshold", type=float, default=0.6)
        parser.add_argument("--negative_threshold", type=float, default=0.45)
        parser.add_argument("--alpha", type=float, default=0.25)
        parser.add_argument("--gamma", type=float, default=2.0)
        parser.add_argument("--smoothl1_beta", type=float, default=1.0)                 # TODO: tune hyperparam
        parser.add_argument("--box_loss_weight", type=float, default=2.0)
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--legacy_focal", type=bool, default=False)
        parser.add_argument("--jit_loss", type=bool, default=False)
        return parent_parser


if __name__ == "__main__":
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser = PointPillars.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PointPillars(args)

    D, P, N = 4, 12000, 100
    x = torch.zeros((1, D, P, N), dtype=torch.float32)
    indices = torch.zeros((1, P, 2), dtype=torch.int32)

    box_out, cls_out = model(x, indices)
    print("cls_out:", cls_out.shape)
    print("box_out:", box_out.shape)
