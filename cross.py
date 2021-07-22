import torch

import itertools, contextlib
from functools import partial

from unet import UNet, ResNet, Encoder, Decoder


class SpatialTransformer(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.net = torch.nn.Sequential( \
                UNet(2*channels, 32, (32, 64, 64, 64, 64)), \
                torch.nn.LeakyReLU(inplace=True), \
                torch.nn.Conv2d(32, 2, kernel_size=3, padding=1))
        with torch.no_grad():
            for param in self.net.parameters():
                param = param / 100.0
        #torch.nn.init.normal_(self.net[-1].weight, 0, 1e-5)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, moving, fixed, features=None):
        theta = torch.Tensor([[[1,0,0],[0,1,0]]]).to(moving, non_blocking=True)
        grid = torch.nn.functional.affine_grid( \
                theta, moving[0:1].shape, align_corners=False)
        self.offset = \
                self.net(torch.cat([moving, fixed], 1)).permute(0, 2, 3, 1)
        self.grid = grid + self.offset
        return self.offset

    def warp(self, img, interp=False):
        warped = torch.nn.functional.grid_sample( \
                img.float(), self.grid.float(), align_corners=False)
        if interp and (img.shape != img.shape):
            warped = torch.nn.functional.interpolate( \
                    warped, size=img.shape[2:])
        return warped


class RecNet(torch.nn.Module):
    def __init__(self, st=True):
        super().__init__()
        self.enable_st = st
        self.net_rec = ResNet(2, 64, 64, 2, res=True)
        self.net_st = SpatialTransformer()
        self.net_enc = Encoder(1, (64, 64, 64, 64))
        self.net_aux = Encoder(1, (64, 64, 64, 64))
        self.net_dec = Decoder(1, (64, 64, 64, 64), (64*2, 64*2, 64*2, 64*2))

    def forward(self, aux, img):
        assert aux.shape == img.shape
        assert len(aux.shape) == 4 #N,C,H,W
        # both aux and img are complex (2-channel) 2d image
        self.aux = aux
        self.aux_abs = torch.norm(self.aux, p=2, dim=1, keepdim=True)
        self.img = img
        self.img_abs = torch.norm(self.img, p=2, dim=1, keepdim=True)
        self.mid = self.net_rec(img) + self.img_abs
        features_mid = self.net_enc(self.mid)
        features_aux = self.net_aux(self.aux_abs)
        if self.enable_st:
            (self.warped, *features_aux), self.offset = self.net_st( \
                    fixed=self.mid, moving=self.aux_abs, features=[self.aux_abs, *features_aux])
        else:
            self.offset = torch.zeros(img.shape[0], *img.shape[2:], 2).to(img)
            self.warped = self.aux_abs.detach()
        bridges = list(map(partial(torch.cat, dim=1), zip(features_mid, features_aux)))
        self.rec = self.net_dec(bridges)
        return self.rec

#class Trans(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.netG = torch.nn.Sequential( \
#                UNet(2, 32, (32, 32, 32, 32, 32)), \
#                torch.nn.LeakyReLU(inplace=True), \
#                torch.nn.Conv2d(32, 2, kernel_size=3, padding=1))
#        self.netD = 

if __name__ == '__main__':
    device = 'cuda'
    net = RecNet()
    sampled_img = torch.rand(3, 2, 256, 256).to(device)
    aux_img = torch.rand(3, 2, 256, 256).to(device)
    net.to(device)
    net(aux_img, sampled_img)
