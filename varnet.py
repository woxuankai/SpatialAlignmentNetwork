"""
Oct 12, 2021
Combined and modified by Kai Xuan <woxuankai@gmail.com>
Code for VarNet was downloaded from github.com/facebookresearch/fastMRI
    with commit 3f9acefc6f740c789e1b720f944ab7821c319226
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

from signal_utils import rss, fft2, ifft2


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    """
    Oct 14, 2021. Kai Xuan
    Note this Unet is NOT designed for complex input/output.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        assert not torch.is_complex(image)
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 1,
        out_chans: int = 1,
        use_ref: bool = False,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
        """
        super().__init__()
        self.use_ref = use_ref
        if self.use_ref:
            self.unet = Unet(
                in_chans=in_chans*3,
                out_chans=out_chans*2,
                chans=chans,
                num_pool_layers=num_pools,
            )
            self.ref_norm = torch.nn.InstanceNorm2d(in_chans)
        else:
            self.unet = Unet(
                in_chans=in_chans*2,
                out_chans=out_chans*2,
                chans=chans,
                num_pool_layers=num_pools,
            )
        self.in_chans = in_chans
        self.out_chans = out_chans

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        return torch.cat([x.real, x.imag], dim=1)

    def chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(x)
        _, c, _, _ = x.shape
        assert c % 2 == 0
        c = c // 2
        return torch.complex(x[:,:c], x[:,c:])

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        assert c%2 == 0
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std + 1e-6), mean, std

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

    def forward(
        self, 
        x: torch.Tensor,
        ref: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        assert len(x.shape) == 4
        assert torch.is_complex(x)
        assert x.shape[1] == self.in_chans

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        if self.use_ref:
            assert not torch.is_complex(ref)
            ref = self.ref_norm(ref)
            ref, _ = self.pad(ref)
            x = torch.cat([x, ref], dim=1)
        else:
            assert ref is None

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_dim_to_complex(x)

        assert x.shape[1] == self.out_chans

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.

    Note SensitivityModel is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 1,
        out_chans: int = 1,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
        )
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.down = nn.AvgPool2d(2)
        #self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear')
        #self.down = lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear')
    '''
    def up(self, x):
        xR, xI = x.real, x.imag
        xR = F.interpolate(xR, scale_factor=2, mode='bilinear')
        xI = F.interpolate(xI, scale_factor=2, mode='bilinear')
        return torch.complex(xR, xI)

    def down(self, x):
        xR, xI = x.real, x.imag
        xR = F.avg_pool2d(xR, 2)
        xI = F.avg_pool2d(xI, 2)
        return torch.complex(xR, xI)
    '''

    def forward(
        self,
        masked_kspace: torch.Tensor,
        num_low_frequencies: int,
    ) -> torch.Tensor:
        # get ACS signals only (i.e. preserve low freq only)
        ACS_mask = torch.ones(masked_kspace.shape[-1])
        ACS_mask[num_low_frequencies:] = 0
        ACS_mask = torch.roll(ACS_mask, -num_low_frequencies//2)
        ACS_mask = ACS_mask[None, None, None, :].to(masked_kspace)
        ACS_kspace = ACS_mask * masked_kspace

        # convert to image space
        ACS_images = ifft2(ACS_kspace)

        # estimate sensitivities independently
        N, C, H, W = ACS_images.shape
        #assert H%2 == 0 and W%2 == 0
        batched_channels = ACS_images.reshape(N*C, 1, H, W)
        #batched_channels = self.down(batched_channels)
        chunk_size = N*2
        chunks = torch.split(batched_channels, chunk_size, dim=0)
        output = []
        for chunk in chunks:
            output.append(self.norm_unet(chunk))
        sensitivity = torch.cat(output, dim=0)
        del output
        #sensitivity = self.norm_unet(batched_channels)
        #sensitivity = self.up(sensitivity)
        sensitivity = sensitivity.reshape(N, C, H, W)
        sensitivity = sensitivity / (rss(sensitivity) + 1e-6)
        return sensitivity

class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        use_ref: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.use_ref = use_ref
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, use_ref=use_ref)) \
                    for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        ref: torch.Tensor,
        num_low_frequencies: int,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        if self.use_ref:
            ref = rss(ref)

        for cascade in self.cascades:
            kspace_pred = cascade(
                current_kspace = kspace_pred,
                ref_kspace = masked_kspace,
                mask = mask,
                sens_maps = sens_maps,
                ref_image = ref)

        return rss(ifft2(kspace_pred))

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fft2(image * sens_maps)

    def sens_reduce(self, kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return (ifft2(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        ref_image: torch.Tensor,
    ) -> torch.Tensor:
        # refinement
        model_term = self.sens_reduce(current_kspace, sens_maps)
        model_term = self.model(model_term, ref_image)
        model_term = self.sens_expand(model_term, sens_maps)
        # soft DC
        zero = torch.zeros(1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        # combine
        return current_kspace - soft_dc - model_term


if __name__ == '__main__':

    varnet = VarNet( \
            num_cascades=8, \
            sens_chans=8, \
            sens_pools=4, \
            chans=18, \
            pools=4, \
            )
    from ssimloss import ssimloss

    size = 320
    sparsity = 0.25
    center = 0.32
    N, C = 3, 1
    masked_kspace = torch.randn(N, C, size, size, dtype=torch.cfloat)
    mask = torch.rand(size)
    mask[:int(size*sparsity*center):] = 2
    mask = torch.roll(mask, -int(size*sparsity*center)//2)
    _, ind = torch.topk(mask, int(sparsity*size))
    mask = torch.zeros(size, dtype=torch.bool).scatter( \
            -1, ind, torch.ones(size, dtype=torch.bool))
    mask = mask[None,None, None, :]
    masked_kspace = mask * masked_kspace
    num_low_frequencies = int(size*sparsity*center)
    varnet, masked_kspace, mask = varnet.cuda(), masked_kspace.cuda(), mask.cuda()
    result = varnet(masked_kspace, mask, masked_kspace.abs(), num_low_frequencies)
    ssimloss(result, ifft2(masked_kspace).abs()).backward()
