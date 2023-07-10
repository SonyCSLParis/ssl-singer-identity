import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from singer_identity.trainer import SSLTrainer
from singer_identity.trainer_byol import BYOL
from singer_identity.data.siamese_encoders import SiameseEncodersDataModule

# TODO verify that this is really needed

# # Comment this line to run the script without callbacks and avoid possible
# # visual package import errors
# from callbacks import *


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", default=None)


if __name__ == "__main__":
    cli = CLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )

    ckpt_path = cli.config["ckpt_path"]

    if ckpt_path is not None:
        step = torch.load(ckpt_path, map_location="cpu")["global_step"]
        cli.trainer.fit_loop.epoch_loop._batches_that_stepped = step

    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)
