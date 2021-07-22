import torch
import numpy as np
from functools import partial

from unet import CatSequential, ResSequential, NullModule

def Down():
    return torch.nn.AvgPool2d(2)

class Conv(torch.nn.Module):
    # norm_layer - activation - weight_norm(conv3x3)
    def __init__(self, in_channels, out_channels, \
            conv = partial(torch.nn.Conv2d, kernel_size=3, padding=1), \
            act = partial(torch.nn.ReLU, inplace=True), \
            norm_layer=torch.nn.BatchNorm2d, \
            weight_norm = torch.nn.utils.spectral_norm, \
            init = torch.nn.init.xavier_normal_):
        super().__init__()
        self.norm_layer = \
                NullModule() if norm_layer is None else norm_layer(in_channels)
        self.act = NullModule() if act is None else act()
        self.conv = conv(in_channels, out_channels)
        init(self.conv.weight)
        self.conv = weight_norm(self.conv)

    def forward(self, x):
        return self.conv(self.act(self.norm_layer(x)))

#def Down(*args, **kw):
#    if len(args) == 0:
#        in_channels, out_channels = kw['in_channels'], kw['out_channels']
#    elif len(args) == 1:
#        in_channels, out_channels = args[0], kw['out_channels']
#    elif len(args) == 2:
#        in_channels, out_channels = args
#    else:
#        assert False, 'Unable to tell in_channels and out_channels'
#    if in_channels == out_channels:
#        return torch.nn.AvgPool2d(2)
#    else:
#        return ConvDow(*args, **kw)

class ConvDown(Conv):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw, \
                conv=partial(torch.nn.Conv2d, kernel_size=2, stride=2))

def Up():
    return torch.nn.Upsample(scale_factor=2, mode='nearest')

#def Up(*args, **kw):
#    if len(args) == 0:
#        in_channels, out_channels = kw['in_channels'], kw['out_channels']
#    elif len(args) == 1:
#        in_channels, out_channels = args[0], kw['out_channels']
#    elif len(args) == 2:
#        in_channels, out_channels = args
#    else:
#        assert False, 'Unable to tell in_channels and out_channels'
#    if in_channels == out_channels:
#        return torch.nn.Upsample(scale_factor=2, mode='nearest')
#    else:
#        return ConvUp(*args, **kw)

def ConvUp(Conv):
        super().__init__(*args, **kw, \
                conv=partial(torch.nn.ConvTranspose2d, kernel_size=2, stride=2))

#def Up(in_channels, out_channels):
#    return torch.nn.Sequential( \
#            torch.nn.Conv2d( \
#            in_channels=in_channels, out_channels=4*out_channels, \
#            kernel_size=1, bias=False)
#            torch.nn.PixelShuffle(upscale_factor=2))

class NetG(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super().__init__()
        layers = list(layers)
        num_convs = 2
        current_layer = layers.pop()
        upper_layer = layers.pop()
        unet = CatSequential( \
                ConvDown(upper_layer, current_layer), \
                ResSequential( \
                *[Conv(current_layer, current_layer) \
                for _ in range(num_convs)]), \
                Up())#ConvUp(current_layer, current_layer))
        for layer in reversed(layers):
            lower_layer, current_layer, upper_layer = \
                    current_layer, upper_layer, layer
            unet = CatSequential( \
                    ConvDown(upper_layer, current_layer) ,\
                    ResSequential( \
                    *[Conv(current_layer, current_layer) \
                    for _ in range(num_convs)]), \
                    unet, \
                    Conv(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    Up())#ConvUp(current_layer, current_layer))
        lower_layer, current_layer = \
                current_layer, upper_layer
        self.unet = torch.nn.Sequential( \
                    Conv(in_channels, current_layer), \
                    ResSequential( \
                    *[Conv(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    unet, \
                    Conv(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    Conv(current_layer, out_channels))

    def forward(self, x):
        return self.unet(x)

class NetD(torch.nn.Module):
    def __init__(self, in_channels, layers):
        super().__init__()
        #assert out_channels == 1
        out_channels = 1
        layers = list(layers)
        num_convs = 2
        current_layer = in_channels
        conv = partial(Conv, norm_layer=None)
        net = []
        for block in layers:
            for layer in block:
                last_layer, current_layer = current_layer, layer
                net.append(conv(last_layer, current_layer))
            net.append(Down())
        net[-1] = conv(layer, out_channels)
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
            
def loss_gan(predict, real=True, D_loss=True):
    if D_loss:
        loss = torch.clamp((-predict if real else predict), min=-1)
    else:
        loss = predict if real else -predict
    assert not(real==True and D_loss==False), 'are you sure?'
    return loss.mean()

if __name__ == '__main__':
    img = torch.randn(5, 1, 320, 320)
    netG = NetG(in_channels=1, out_channels=1, layers=(64, 128, 256, 256))
    netD = NetD(in_channels=1, layers=([64]*3, [128]*3, [256]*3, [512]*3, [1024]*3, [1024]*3))
    out = netG(img)
    print(out.shape)
    out = netD(out)
    print(out.shape)
