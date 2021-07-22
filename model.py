import os, sys, contextlib, time
import numpy as np
import torch
import torch.fft

from masks import Mask, StandardMask, RandomMask, LowpassMask, EquispacedMask, LOUPEMask, TaylorMask
from basemodel import BaseModel, Config
import metrics
from metrics import mi as metrics_mi
import lnccloss
#from miloss import ms_mi_loss as sim_loss
#from mineloss import MineLossPatch

from cross import SpatialTransformer
from gan import loss_gan, NetD, NetG
from unet import ResNet


def gradient_loss(s):
    assert s.shape[-1] == 2, 'not 2D grid?'
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dy = torch.abs(s[:, 1:, :, :] - s[:, :-1, :, :])
    dy = dy*dy
    dx = dx*dx
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.0

def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifftn(x, dim=(-2, -1), norm='ortho')
    return x

def fftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2]//2, x.shape[-1]//2), dims=(-2, -1))
    return x

def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2]+1)//2, (x.shape[-1]+1)//2), dims=(-2, -1))
    return x

def rss(x):
    assert len(x.shape) == 4
    if torch.is_complex(x):
        x = x.abs()
    return torch.norm(x, p=2, dim=1, keepdim=True)

rec_losses = { \
        'MAE': torch.nn.functional.l1_loss, \
        'SSIM': lambda x, y: -ssimloss.ssim(x, y)}

mask_losses = {'MASK_MAE': \
        lambda x: torch.nn.functional.l1_loss(x, torch.zeros_like(x))}

#metrics = {'PSNR': metrics.psnr, 'SSIM': metrics.ssim, \
#        'MAE': metrics.mae, 'MSE': metrics.mse}#, 'MI': metrics.mi}

masks = {"mask": Mask,
        "taylor": TaylorMask,
        "standard": StandardMask,
        #"random": RandomMask,
        "lowpass": LowpassMask,
        "equispaced": EquispacedMask,
        "loupe": LOUPEMask}


class CSModel(BaseModel):
    def build(self, cfg):
        super().build(cfg)
        sparsity = cfg.sparsity
        shape = cfg.shape
        mask_lr = cfg.mask_lr
        mask = cfg.mask
        #if 'wd' in cfg:
        #    self.cfg.wd = cfg.wd
        #else:
        #    self.cfg.wd = 0
        self.cfg.weight_gan = 0.1
        self.cfg.weight_gan_sim = 1.0
        self.cfg.weight_sim = 1.0
        #self.cfg.wd = cfg.wd
        if 'coils' in self.cfg:
            coils = self.cfg.coils
        else:
            coils = 1
        if 'smooth' in cfg:
            self.cfg.weight_smooth = cfg.smooth
        else:
            self.cfg.weight_smooth = 100
        assert cfg.lr == 1e-4
        if mask in ["mask", "taylor"]:
            self.net_mask = masks[mask](shape)
        else:
            self.net_mask = masks[mask](sparsity, shape)
        #self.sim_net = MineLossPatch()
        self.net_G = NetG(in_channels=1, out_channels=1, \
                layers=(64, 128, 256, 512, 512))
        self.net_D = NetD(in_channels=2, \
                layers=([64]*2, [128]*2, [256]*2, [256]*2, [256]*2))
        self.net_T = SpatialTransformer(channels=coils)
        self.net_R = ResNet(3*coils, 1, [96]*4+[64]*4+[32]*4+[16]*4, res=True)
        self.optim_G = torch.optim.AdamW(self.net_G.parameters(), \
                lr=cfg.lr, weight_decay=0)
        self.optim_D = torch.optim.AdamW(self.net_D.parameters(), \
                lr=cfg.lr, weight_decay=0)
        self.optim_T = torch.optim.AdamW(self.net_T.parameters(), \
                lr=cfg.lr, weight_decay=0)
        self.optim_R = torch.optim.AdamW(self.net_R.parameters(), \
                lr=cfg.lr, weight_decay=0)
        self.optim_M = torch.optim.AdamW(self.net_mask.parameters(), \
                lr=cfg.lr, weight_decay=0)
        #assert self.cfg.reg in ('None', 'Rec', 'Mixed', 'GAN-Only')
        self.use_amp = False
        self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def set_input(self, img_full, img_aux=None):
        # reset environment
        if_val = lambda x: x.startswith(('loss_', 'img_', 'metric_'))
        for val_name in list(filter(if_val, self.__dict__.keys())):
            delattr(self, val_name)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self.img_full = img_full
            if img_aux is None:
                self.img_aux = torch.zeros_like(img_full)
            else:
                self.img_aux = img_aux
            self.img_k_full = fft2(self.img_full)
            with torch.no_grad(): # avoid update of mask
                #self.img_k_sampled = self.net_mask(self.img_k_full)
                self.img_k_sampled = self.img_k_full * (1-self.net_mask.pruned.float())
            self.img_sampled = ifft2(self.img_k_sampled)
            self.img_full_rss = rss(self.img_full)
            self.img_sampled_rss = rss(self.img_sampled)
            self.img_aux_rss = rss(self.img_aux)
            #mask = torch.ones(self.cfg.shape).to(self.net_mask.pruned, non_blocking=True)
            #mask.masked_scatter_(self.net_mask.pruned, torch.zeros_like(mask))
            with torch.no_grad():
                self.img_mask = fftshift2(torch.ones_like(self.img_full_rss) - self.net_mask.pruned.float())

    def forwardG(self):
        # modality translation
        aux_TR, aux_RT = torch.chunk( \
                self.img_aux_rss, 2, dim=0)
        T = self.net_G(aux_RT)
        R, RT = torch.chunk(self.net_T.warp(torch.cat((aux_TR,T))), 2)
        TR = self.net_G(R)
        self.img_synth = torch.cat((R, T), dim=0)
        self.img_aligned = torch.cat((TR, RT), dim=0)
        # loss
        self.loss_gan_sim = torch.nn.functional.l1_loss( \
                self.img_aligned, self.img_full_rss)
        self.loss_all += self.loss_gan_sim * self.cfg.weight_gan_sim

    def forwardT(self):
        # translation
        self.img_offset = self.net_T( \
        #        moving = self.img_aux_rss, fixed = self.img_sampled_rss)
                moving = self.img_aux.abs(), fixed = self.img_sampled.abs())
        self.img_warped = self.net_T.warp(self.img_aux.abs()) #self.img_aux
        self.img_warped_rss = rss(self.img_warped)
        # loss
        self.loss_smooth = gradient_loss(self.img_offset)
        self.loss_all += self.loss_smooth * self.cfg.weight_smooth

    def forwardR(self):
        self.img_rec = self.net_R(torch.cat((self.img_warped, \
                self.img_sampled.real, self.img_sampled.imag), dim=1))
        # loss
        self.loss_sim = torch.nn.functional.l1_loss( \
                self.img_full_rss, self.img_rec)
        self.loss_all += self.loss_sim * self.cfg.weight_sim

    def forwardD(self, D_loss):
        fake = torch.cat( \
                (self.img_aligned, self.img_aux_rss), dim=1)
        real = torch.cat( \
                (self.img_full_rss, self.img_aux_rss), dim=1)
        if D_loss:
            self.loss_gan_Dfake = loss_gan( \
                    self.net_D(fake.detach()), real=False, D_loss=True)
            self.loss_gan_Dreal = loss_gan( \
                    self.net_D(real.detach()), real=True, D_loss=True)
            self.loss_all += (self.loss_gan_Dfake + self.loss_gan_Dreal) \
                    * self.cfg.weight_gan
        else:
            self.loss_gan_G = loss_gan( \
                    self.net_D(fake), real=False, D_loss=False)
            self.loss_all += self.loss_gan_G*self.cfg.weight_gan


    def update(self):
        assert self.training == True
        if self.cfg.reg == 'None':
            # reconstruciton only
            self.loss_all = 0
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                with torch.no_grad():
                    self.forwardT()
                self.loss_all = 0
                self.forwardR()
            self.optim_R.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_R)
            self.scalar.update()
        elif self.cfg.reg == 'Rec':
            # reconstruction and rec-guided registration
            self.loss_all = 0
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                self.forwardT()
                self.forwardR()
            self.optim_T.zero_grad()
            self.optim_R.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_T)
            self.scalar.step(self.optim_R)
            self.scalar.update()
            #print(t4-t0, t4-t3, t3-t2, t2-t1, t1-t0)
        elif self.cfg.reg == 'Mixed':
            # reconstruction and GAN-guided registration
            # update T, G, and R
            self.loss_all = 0
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                self.forwardT()
                self.forwardG()
                self.forwardR()
                self.forwardD(D_loss=False)
            self.optim_T.zero_grad()
            self.optim_G.zero_grad()
            self.optim_R.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_T)
            self.scalar.step(self.optim_G)
            self.scalar.step(self.optim_R)
            # self.scalar.update()
            # update D
            self.loss_all = 0#torch.tensor(0, dtype=torch.float)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                self.forwardD(D_loss=True)
            self.optim_D.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_D)
            # self.scalar.update()
        elif self.cfg.reg == 'GAN-Only':
            # GAN-guided registration only
            # update T and G
            self.loss_all = 0
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                self.forwardT()
                self.forwardG()
                self.forwardD(D_loss=False)
            self.optim_T.zero_grad()
            self.optim_G.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_T)
            self.scalar.step(self.optim_G)
            # self.scalar.update()
            # update D
            self.loss_all = 0#torch.tensor(0, dtype=torch.float)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                self.forwardD(D_loss=True)
            self.optim_D.zero_grad()
            self.scalar.scale(self.loss_all).backward()
            self.scalar.step(self.optim_D)
            self.scalar.update()
        else:
            assert False
        del self.loss_all

    def test(self):
        assert self.training == False
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.no_grad():
                self.loss_all = 0
                self.forwardT()
                self.loss_all = 0
                self.forwardG()
                self.loss_all = 0
                self.forwardR()
                self.metric_MI = metrics_mi(self.img_full_rss, self.img_warped_rss)
                self.metric_PSNR = metrics.psnr(self.img_full_rss, self.img_rec)
                self.metric_SSIM = metrics.ssim(self.img_full_rss, self.img_rec)
                self.metric_MAE = metrics.mae(self.img_full_rss, self.img_rec)
                self.metric_MSE = metrics.mse(self.img_full_rss, self.img_rec)
                if self.cfg.reg == 'GAN-Only':
                    returnVal = -self.metric_MI
                else:
                    returnVal = self.loss_all.cpu().item()
                del self.loss_all
        return returnVal # return reconstruciton loss

    def prune(self, *args, **kwargs):
        assert False, 'Take care of amp'
        return self.net_mask.prune(*args, **kwargs)

    def get_vis(self, content=None):
        assert content in [None, 'scalars', 'histograms', 'images']
        vis = {}
        if content == 'scalars' or content is None:
            vis['scalars'] = {}
            for loss_name in filter( \
                    lambda x: x.startswith('loss_'), self.__dict__.keys()):
                loss_val = getattr(self, loss_name)
                if loss_val is not None:
                    vis['scalars'][loss_name] = loss_val.item()
            for metric_name in filter( \
                    lambda x: x.startswith('metric_'), self.__dict__.keys()):
                metric_val = getattr(self, metric_name)
                if metric_val is not None:
                    vis['scalars'][metric_name] = metric_val
        if content == 'images' or content is None:
            vis['images'] = {}
            for image_name in filter( \
                    lambda x: x.startswith('img_'), self.__dict__.keys()):
                image_val = getattr(self, image_name)
                if (image_val is not None) \
                        and (image_val.shape[1]==1 or image_val.shape[1]==3) \
                        and not torch.is_complex(image_val):
                    vis['images'][image_name] = image_val.detach()
        if content == 'histograms' or content is None:
            vis['histograms'] = {}
            if self.net_mask.weight is not None:
                vis['histograms']['weights'] = { \
                        'values': self.net_mask.weight.detach()}
        return vis


if __name__ == '__main__':
    cfg = Config()
    cfg.sparsity = 1.0/8
    cfg.lr = 0.001
    cfg.mask_lr = 0.001
    cfg.st = True
    cfg.shape = 256
    cfg.mask = 'mask'
    net = CSModel(cfg)
    device = 'cuda'
    full_img = torch.rand(3, 2, 256, 256).to(device)
    aux_img = torch.rand(3, 2, 256, 256).to(device)
    net.to(device)
    net.train()
    net.set_input(img_full=full_img, img_aux=aux_img)
    net.update()
    print(net.get_vis())
    net.eval()
    net.set_input(img_full=full_img, img_aux=aux_img)
    net.test()
    print(net.get_vis())


