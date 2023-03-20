import math
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from .automap import AutoMap
from .unet import Unet
from .nestedunet import NestedUnet
from .attentionunet import AttentionUnet
from .swinunet import SwinUnet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        backbone_type: str = "unet",
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        if backbone_type == "unet":
            self.unet = Unet(
                in_chans=in_chans,
                out_chans=out_chans,
                chans=chans,
                num_pool_layers=num_pools,
                drop_prob=drop_prob,
            )
        elif backbone_type == "nestedunet":
            self.unet = NestedUnet(
                in_chans=in_chans,
                out_chans=out_chans,
                chans=chans,
                deepsupervision=False,
            )
        elif backbone_type == "attentionunet":
            self.unet = AttentionUnet(
                in_chans=in_chans,
                out_chans=out_chans,
                chans=chans,
            )
        elif backbone_type == "swinunet":
            self.unet = SwinUnet(
                in_chans=in_chans,
                out_chans=out_chans,
                chans=chans,
            )
        else:
            raise ValueError(f"unrecognized type {backbone_type}")

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multi-channel k-space data and then a U-Net to the coil images to estimate coil sensitivities. 
    It can be used with the concat variational network.

    TODO: try to mod something like self-attention or attention gate in unet.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class ResidualVarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net regularizer. 
    To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        organize_type: str = "cascade",
        residual_type : str = "add",
        backbone_type: str = "unet",
        transform_type: str = "fourier",
        consistency_type: str = "soft",
        num_blocks: int = 3,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            organize_type: Type of blocks organization in variational network.
            residual_type: Type of residual path between VarNet blocks.
            backbone_type: Type of backbone in variational network.
            transform_type: Type of transformation between kspace and reconstruction in variational network.
            consistency_type: Type of data consistency for K-Space.
            num_blocks: Number of blocks in variational network.
            sens_chans: Number of channels for U-Net in sensitivity model.
            sens_pools: Number of downsampling and upsampling layers for U-Net in sensitivity model.
            chans: Number of channels for U-Net in blocks.
            pools: Number of downsampling and upsampling layers for U-Net in blocks.
            mask_center: Whether to mask center of k-space for sensitivity map calculation.
        """
        super().__init__()
        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools, mask_center=mask_center)
        
        self.fft, self.ifft = self.get_transform_op(transform_type)
        
        # if organize_type == "recurrent":
        #     recurrent_block = VarNetBlock(NormUnet(chans, pools, type=backbone_type), self.fft, self.ifft)
        #     self.blocks = nn.ModuleList([recurrent_block for _ in range(num_blocks)])
        # elif organize_type == "recurrentv2":
        #     self.block = VarNetBlock(NormUnet(chans, pools, type=backbone_type), self.fft, self.ifft)
        # else:
        #     self.blocks = nn.ModuleList([VarNetBlock(NormUnet(chans, pools, type=backbone_type), self.fft, self.ifft) for _ in range(num_blocks)])

        # if organize_type == "cascade" or "residual" or "residualv2" or "residualv3" or "residualv4":
        # if organize_type == "cascade" or "residual_sum" or "residual_mean" or "residual_sum_inputless" or "residual_mean_inputless":
        if organize_type == "cascade":
            self.blocks = nn.ModuleList([VarNetBlock(
                    NormUnet(chans, pools, backbone_type=backbone_type), self.fft, self.ifft, consistency_type=consistency_type
                    ) for _ in range(num_blocks)])
        elif organize_type == "recurrent":
            self.block == VarNetBlock(NormUnet(chans, pools, backbone_type=backbone_type), self.fft, self.ifft, consistency_type=consistency_type)
        else:
            raise ValueError(f"unrecognized type {organize_type}")

        self.organize_type = organize_type
        self.residual_type = residual_type

        self.num_blocks = num_blocks

    def get_transform_op(self, transform_type: str):
        if transform_type == "fourier":
            return fastmri.fft2c, fastmri.ifft2c
        elif transform_type == "automap":
            return AutoMap(mode="fft"), AutoMap(mode="ifft")
        elif transform_type == "direct":
            return nn.Identity, nn.Identity
        else:
            raise ValueError(f"unrecognized type {transform_type}")

    # def get_current_recon(self, x: torch.Tensor, sens_maps: torch.Tensor):
    #     return fastmri.complex_abs(fastmri.complex_mul(self.ifft(x), fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)).squeeze(1)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        
        # extra_outputs = defaultdict(list)
        # extra_outputs["masks"].append(mask.detach().cpu())

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        # extra_outputs["sense"].append(sens_maps.detach().cpu())
        
        # kspace_pred = masked_kspace.clone()
        kspace_preds = [masked_kspace.clone()]
        # current_recon = self.get_current_recon(kspace_pred, sens_maps)
        # extra_outputs["recons"].append(current_recon.detach().cpu())
        
        for idx in range(self.num_blocks):
            if self.organize_type == "cascade" and self.residual_type == "none":
                kspace_preds.append(self.blocks[idx](kspace_preds[-1], masked_kspace, mask, sens_maps))
            elif self.organize_type == "cascade" and self.residual_type == "add":
                kspace_preds.append(torch.stack((self.blocks[idx](kspace_preds[-1], masked_kspace, mask, sens_maps), kspace_preds[-1])).sum(dim=0))
            elif self.organize_type == "cascade" and self.residual_type == "mean":
                kspace_preds.append(torch.stack((self.blocks[idx](kspace_preds[-1], masked_kspace, mask, sens_maps), kspace_preds[-1])).mean(dim=0))
            
            elif self.organize_type == "recurrent" and self.residual_type == "none":
                kspace_preds.append(self.block(kspace_preds[-1], masked_kspace, mask, sens_maps))
            elif self.organize_type == "recurrent" and self.residual_type == "add":
                kspace_preds.append(torch.stack((self.block(kspace_preds[-1], masked_kspace, mask, sens_maps), kspace_preds[-1])).sum(dim=0))
            elif self.organize_type == "recurrent" and self.residual_type == "mean":
                kspace_preds.append(torch.stack((self.block(kspace_preds[-1], masked_kspace, mask, sens_maps), kspace_preds[-1])).mean(dim=0))

            else:
                raise ValueError(f"unrecognized types {self.organize_type, self.residual_type}")
            
        # for block in self.blocks:
        #     if self.organize_type == "cascade":
        #         kspace_pred = block(kspace_pred, masked_kspace, mask, sens_maps)
        #     elif self.organize_type == "residual":
        #         kspace_pred = torch.stack((block(kspace_pred, masked_kspace, mask, sens_maps), kspace_pred)).sum(dim=0)
        #     elif self.organize_type == "residualv2":
        #         kspace_pred = torch.stack((block(kspace_pred, masked_kspace, mask, sens_maps), kspace_pred)).mean(dim=0)
        #     elif self.organize_type == "recurrent":  
        #         kspace_pred = block(kspace_pred, masked_kspace, mask, sens_maps)
        #     else:
        #         raise ValueError(f"unrecognized type {self.organize_type}")
        
        # if self.organize_type == "recurrentv2":
        #     for _ in range(self.num_blocks):
        #         kspace_pred = torch.stack((self.block(kspace_pred, masked_kspace, mask, sens_maps), kspace_pred)).mean(dim=0)
        
        # if self.organize_type == "residualv3":
        #     kspace_pred_0 = self.blocks[0](kspace_pred, masked_kspace, mask, sens_maps)
        #     kspace_pred_1 = torch.stack((self.blocks[1](kspace_pred_0, masked_kspace, mask, sens_maps), kspace_pred_0)).sum(dim=0)
        #     kspace_pred = torch.stack((self.blocks[2](kspace_pred_1, masked_kspace, mask, sens_maps), kspace_pred_1)).sum(dim=0)
            
        # if self.organize_type == "residualv4":
        #     kspace_pred_0 = self.blocks[0](kspace_pred, masked_kspace, mask, sens_maps)
        #     kspace_pred_1 = torch.stack((self.blocks[1](kspace_pred_0, masked_kspace, mask, sens_maps), kspace_pred_0)).mean(dim=0)
        #     kspace_pred = torch.stack((self.blocks[2](kspace_pred_1, masked_kspace, mask, sens_maps), kspace_pred_1)).mean(dim=0)

        # current_recon = self.get_current_recon(kspace_pred, sens_maps)
        # extra_outputs["recons"].append(current_recon.detach().cpu())

        output = fastmri.rss(fastmri.complex_abs(self.ifft(kspace_preds[-1])), dim=1)

        # return output, extra_outputs
        return output


class VarNetBlock(nn.Module):
    """
    Model block for variational network.

    This model applies a combination of soft data consistency with the input model as a regularizer. 
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(self, model: nn.Module, fft, ifft, consistency_type="soft"):
        """
        Args:
            model: Module for "regularization" component of variational network.
        """
        super().__init__()

        self.model = model
        self.fft = fft
        self.ifft = ifft

        # self.consis_weight = nn.Parameter(torch.ones(1)) if consistency_type == "soft" else 1
        if consistency_type == "soft":
            self.consis_weight = nn.Parameter(torch.ones(1))
        elif consistency_type == "hard":
            self.consis_weight == 1
        elif consistency_type == "none":
            self.consis_weight == 0

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return self.fft(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(self.ifft(x), fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        data_consis = torch.where(mask, current_kspace - ref_kspace, zero) * self.consis_weight
        
        model_term = self.sens_expand(self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps)

        return current_kspace - data_consis - model_term

