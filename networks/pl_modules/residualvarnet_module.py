from argparse import ArgumentParser
from collections import defaultdict

import math
import sys
import torch
import numpy as np

import fastmri
from fastmri import evaluate
from fastmri.data import transforms
from fastmri.data.transforms import VarNetSample
from fastmri.pl_modules import MriModule

sys.path.append('.')
from networks.models import ResidualVarNet
from utils import DistributedArraySum, DistributedMetricSum


class ResidualVarNetModule(MriModule):
    """
    Residual VarNet training module.
    """

    def __init__(
        self,
        organize_type: str = "cascade",
        residual_type : str = "add",
        backbone_type: str = "unet",
        transform_type: str = "fourier",
        consistency_type: str = "soft",
        num_blocks: int = 3,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.01,
        # lr_step_size: int = 40,
        # lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        max_epochs: int = 50,
        warmup_epochs: int = 3,
        **kwargs,
    ):
        """
        Args:
            organize_type: Type of blocks organization in variational network.
            residual_type: Type of residual path between VarNet blocks.
            backbone_type: Type of backbone in variational network.
            transform_type: Type of transformation between kspace and reconstruction in variational network.
            consistency_type: Type of data consistency for K-Space.
            num_blocks: Number of blocks in variational network.
            pools: Number of downsampling and upsampling layers for U-Net in blocks.
            chans: Number of channels for U-Net in blocks.
            sens_pools: Number of downsampling and upsampling layers for U-Net in sensitivity model.
            sens_chans: Number of channels for U-Net in sensitivity model.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.organize_type = organize_type
        self.residual_type = residual_type
        self.backbone_type = backbone_type
        self.transfrom_type = transform_type
        self.consistency_type = consistency_type
        self.num_blocks = num_blocks
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        # self.lr_step_size = lr_step_size
        # self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        self.TrainNMSE = DistributedMetricSum()
        self.TrainSSIM = DistributedMetricSum()
        self.TrainPSNR = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.TrainTotExamples = DistributedMetricSum()
        self.TrainTotSliceExamples = DistributedMetricSum()
        self.TrainCondEnt = DistributedMetricSum()
        self.TrainMargDist = DistributedArraySum()

        self.ValNMSE = DistributedMetricSum()
        self.ValSSIM = DistributedMetricSum()
        self.ValPSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.ValTotExamples = DistributedMetricSum()
        self.ValTotSliceExamples = DistributedMetricSum()
        self.ValCondEnt = DistributedMetricSum()
        self.ValMargDist = DistributedArraySum()

        self.varnet = ResidualVarNet(
            organize_type=self.organize_type,
            residual_type=self.residual_type,
            backbone_type=self.backbone_type,
            transform_type=self.transfrom_type,
            consistency_type=self.consistency_type,
            num_blocks=self.num_blocks,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.varnet(masked_kspace, mask)
                        
    def training_step(self, batch, batch_idx):
        # output, extra_outputs = self(batch.masked_kspace, batch.mask)
        output = self(batch.masked_kspace, batch.mask)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
        # self.log("training_loss", loss)
        # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # self.log("learning_rate", lr)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "loss": loss,
            # "lr": lr,
            # "extra_outputs": extra_outputs,
        }

    def training_step_end(self, train_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "loss",
            # "lr",
            # "extra_outputs",
        ):
            if k not in train_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by training_step."
                )
        if train_logs["output"].ndim == 2:
            train_logs["output"] = train_logs["output"].unsqueeze(0)
        elif train_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")
        if train_logs["target"].ndim == 2:
            train_logs["target"] = train_logs["target"].unsqueeze(0)
        elif train_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from training_step.")

        # compute evaluation metrics
        mse_values = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_values = defaultdict(dict)
        max_values = dict()
        for i, fname in enumerate(train_logs["fname"]):
            slice_num = int(train_logs["slice_num"][i].cpu())
            maxval = train_logs["max_value"][i].cpu().numpy()
            output = train_logs["output"][i].detach().cpu().numpy()
            target = train_logs["target"][i].cpu().numpy()

            mse_values[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_values[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_values[fname] = maxval

        return {
            "loss": train_logs["loss"],
            "mse_values": mse_values,
            "target_norms": target_norms,
            "ssim_values": ssim_values,
            "max_values": max_values,
        }

    def training_epoch_end(self, train_logs):
        losses = []
        mse_values = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_values = defaultdict(dict)
        max_values = dict()

        # use dict updates to handle duplicate slices
        for train_log in train_logs:
            losses.append(train_log["loss"].data.view(-1))

            for k in train_log["mse_values"].keys():
                mse_values[k].update(train_log["mse_values"][k])
            for k in train_log["target_norms"].keys():
                target_norms[k].update(train_log["target_norms"][k])
            for k in train_log["ssim_values"].keys():
                ssim_values[k].update(train_log["ssim_values"][k])
            for k in train_log["max_values"]:
                max_values[k] = train_log["max_values"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_values.keys()
            == target_norms.keys()
            == ssim_values.keys()
            == max_values.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_values.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_values[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_values[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_values[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.TrainNMSE(metrics["nmse"])
        metrics["ssim"] = self.TrainSSIM(metrics["ssim"])
        metrics["psnr"] = self.TrainPSNR(metrics["psnr"])
        tot_examples = self.TrainTotExamples(torch.tensor(local_examples))
        train_loss = self.TrainLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TrainTotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("learning_rate", self.optimizer.state_dict()['param_groups'][0]['lr'], prog_bar=True)
        self.log("train_loss", train_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"train_metrics/{metric}", value / tot_examples, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # batch: VarNetSample
        # output, extra_outputs = self.forward(batch.masked_kspace, batch.mask)
        output = self.forward(batch.masked_kspace, batch.mask)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.loss(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)
        # self.log("val_loss", loss)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": loss,
            # "extra_outputs": extra_outputs,
        }

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_values = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_values = defaultdict(dict)
        max_values = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_values[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_values[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_values[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_values": mse_values,
            "target_norms": target_norms,
            "ssim_values": ssim_values,
            "max_values": max_values,
        }

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_values = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_values = defaultdict(dict)
        max_values = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_values"].keys():
                mse_values[k].update(val_log["mse_values"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_values"].keys():
                ssim_values[k].update(val_log["ssim_values"][k])
            for k in val_log["max_values"]:
                max_values[k] = val_log["max_values"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_values.keys()
            == target_norms.keys()
            == ssim_values.keys()
            == max_values.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_values.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_values[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_values[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_values[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.ValNMSE(metrics["nmse"])
        metrics["ssim"] = self.ValSSIM(metrics["ssim"])
        metrics["psnr"] = self.ValPSNR(metrics["psnr"])
        tot_examples = self.ValTotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.ValTotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # output, extra_outputs = self(batch.masked_kspace, batch.mask)
        output = self(batch.masked_kspace, batch.mask)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        # self.optim = torch.optim.Adam(
        #     self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        self.optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optim, self.lr_step_size, self.lr_gamma
        # )
        cosinelr_warmup = lambda epoch: epoch / self.warmup_epochs if epoch <= self.warmup_epochs else 0.5 * (math.cos((epoch - self.warmup_epochs) /(self.max_epochs - self.warmup_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=cosinelr_warmup
        )

        return [self.optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_blocks",
            default=3,
            type=int,
            help="Number of VarNet blocks",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--max_epochs", default=50, type=int, help="Number of total epochs"
        )
        parser.add_argument(
            "--warmup_epochs", default=3, type=int, help="Number of warmup epochs at beginning"
        )
        parser.add_argument(
            "--lr", default=0.001, type=float, help="Learning rate in Adam or AdamW"
        )
        # parser.add_argument(
        #     "--lr_step_size",
        #     default=40,
        #     type=int,
        #     help="Epoch at which to decrease step size if stepLR",
        # )
        # parser.add_argument(
        #     "--lr_gamma",
        #     default=0.1,
        #     type=float,
        #     help="Extent to which step size should be decreased",
        # )
        parser.add_argument(
            "--weight_decay",
            default=0.01,
            type=float,
            help="Strength of weight decay regularization in Adam or AdamW",
        )

        return parser
