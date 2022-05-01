from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb

from data.dataset import KittiDataModule
from model import PointPillars


# program-level args
parser = ArgumentParser()
parser.add_argument("data_path", type=str)

# model-specific args
parser = PointPillars.add_model_specific_args(parser)

# trainer-specific args (pass in `--max_epochs 160`)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

wandb.init(config=args, project="popi")

model = PointPillars(args)
dm = KittiDataModule(args.data_path, args)

trainer = pl.Trainer.from_argparse_args(args)
trainer.fit(model, datamodule=dm)
