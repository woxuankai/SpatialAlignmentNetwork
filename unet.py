import torch
from functools import partial
#import torch.nn as nn
#import torch.nn.functional as F

class CatSequential(torch.nn.Module):
    def __init__(self, *modules, dim=1):
        super().__init__()
        self.module = torch.nn.Sequential(*modules)
        self.dim = dim

    def forward(self, x):
        return torch.cat([self.module(x), x], self.dim)

class ResSequential(torch.nn.Module):
    def __init__(self, *modules, sample=None):
        super().__init__()
        self.subnet = torch.nn.Sequential(*modules)
        self.sample = sample

    def forward(self, x):
        out = self.subnet(x)
        x = self.sample(x) if self.sample is not None else x
        return x + out

class NullModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, layers, norm = NullModule):
        super().__init__()
        layers = list(layers)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        act = partial(torch.nn.LeakyReLU, inplace=True)
        #norm = partial(torch.nn.BatchNorm2d) 
        norm = NullModule 
        conv = partial(torch.nn.Conv2d, \
                kernel_size=kernel_size, padding=padding)
        conv_norm_act = lambda in_ch, out_ch: \
                torch.nn.Sequential(conv(in_ch, out_ch), norm(out_ch), act())
        resblock = lambda channels: \
                ResSequential(*[conv_norm_act(channels, channels) \
                for _ in range(num_convs)])
        down = partial(torch.nn.AvgPool2d, kernel_size=2, stride=2)

        # uppest (change channels, resblock)
        current_layer = layers[0]
        self.encoders = torch.nn.ModuleList([torch.nn.Sequential( \
                conv_norm_act(in_channels, current_layer), \
                resblock(current_layer))])
        # middle (change size, change channels, resblock)
        for layer in layers[1:-1]:
            current_layer, upper_layer = layer, current_layer
            self.encoders.append(torch.nn.Sequential( \
                    down(), conv_norm_act(upper_layer, current_layer), \
                    resblock(current_layer)))
        # lowest (change shape and channels only)
        self.encoders.append(torch.nn.Sequential( \
                down(), conv_norm_act(layers[-2], layers[-1])))

    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        return features

class Decoder(torch.nn.Module):
    def __init__(self, out_channels, layers, bridges, norm = NullModule):
        super().__init__()
        layers = list(layers)
        bridges = list(bridges)
        assert len(layers) == len(bridges)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        act = partial(torch.nn.LeakyReLU, inplace=True)
        #norm = partial(torch.nn.BatchNorm2d) 
        conv = partial(torch.nn.Conv2d, \
                kernel_size=kernel_size, padding=padding)
        conv_norm_act = lambda in_ch, out_ch: \
                torch.nn.Sequential(conv(in_ch, out_ch), norm(out_ch), act())
        resblock = lambda channels: \
                ResSequential(*[conv_norm_act(channels, channels) \
                for _ in range(num_convs)])
        up = partial(torch.nn.Upsample, scale_factor=(2, 2))

        # lowest (change channels, resblock, change shape)
        bridge, current_layer = bridges[-1], layers[-1]
        self.decoders = torch.nn.ModuleList([torch.nn.Sequential( \
                conv_norm_act(bridge, current_layer), \
                resblock(current_layer), up())])
        # middle (concatenate*, change channels, resblock, change shape)
        for bridge, current_layer, lower_layer in \
                zip(bridges[-2:0:-1], layers[-2:0:-1], layers[:1:-1]):
            self.decoders.append(torch.nn.Sequential( \
                    conv_norm_act(bridge+lower_layer, current_layer), \
                    resblock(current_layer), up()))
        # uppest (concatenate*, change channels, resblock, change channels)
        bridge, current_layer, lower_layer = bridges[0], layers[0], layers[1]
        self.decoders.append(torch.nn.Sequential( \
                conv_norm_act(bridge+lower_layer, current_layer), \
                resblock(current_layer),
                conv(current_layer, out_channels)))

    def forward(self, bridges):
        x = torch.tensor((), dtype=bridges[0].dtype).to(bridges[0])
        for decoder, bridge in zip(self.decoders, bridges[::-1]):
            x = torch.cat([x, bridge], dim=1)
            x = decoder(x)
        return x

def Conv2d(in_channels, out_channels):
    kernel_size = 3
    padding = kernel_size // 2
    return torch.nn.Sequential( \
            torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))

def Up(in_channels, out_channels):
    return torch.nn.Sequential( \
            torch.nn.Upsample(scale_factor=(2,2)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))

def Down(in_channels, out_channels):
    return torch.nn.Sequential( \
            torch.nn.AvgPool2d(2, stride=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))



class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super().__init__()
        layers = list(layers)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        current_layer = layers.pop()
        upper_layer = layers.pop()
        unet = CatSequential( \
                Down(upper_layer, current_layer), \
                ResSequential( \
                *[Conv2d(current_layer, current_layer) \
                for _ in range(num_convs)]), \
                Up(current_layer, current_layer))
        for layer in reversed(layers):
            lower_layer, current_layer, upper_layer = \
                    current_layer, upper_layer, layer
            unet = CatSequential( \
                    Down(upper_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs)]), \
                    unet, \
                    Conv2d(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    Up(current_layer, current_layer))
        lower_layer, current_layer = \
                current_layer, upper_layer
        self.unet = torch.nn.Sequential( \
                    Conv2d(in_channels, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    unet, \
                    Conv2d(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    torch.nn.Conv2d(current_layer, out_channels, \
                    kernel_size, padding=padding))

    def forward(self, x):
        return self.unet(x)


def conv3x3(in_channels, out_channels):
    kernel_size = 3
    padding = kernel_size // 2
    return torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding)

def conv1x1(in_channels, out_channels):
    kernel_size = 1
    padding = 0
    return torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding)

def ResNet(in_channels, out_channels, channels=[64]*4, res=False):
    net = []
    last = channels[0]
    for current in channels[1:]:
        sample = conv1x1(last, current) if last!=current else None
        net = net + [torch.nn.LeakyReLU(inplace=True), \
                ResSequential( \
                    conv3x3(last, current), \
                    torch.nn.LeakyReLU(inplace=True), \
                    conv3x3(current, current), \
                    sample=sample)]
        last = current
    if res:
        sample = conv1x1(channels[0], channels[-1]) \
                if channels[0]!=channels[-1] else None
        net = [ResSequential(*net, sample=sample)]
    net = [conv3x3(in_channels, channels[0]), \
            *net, \
            torch.nn.LeakyReLU(inplace=True),
            conv3x3(channels[-1], out_channels)]
    return torch.nn.Sequential(*net)

if __name__ == '__main__':
    x = torch.randn(5, 4, 320, 320).cuda()
    #net = UNet(3, 2, (64, 64*2, 64*4, 64*8, 64*16)).cuda()
    net = UNet(4, 2, (64, 64*2, 64*4)).cuda()
    y = net(x)
    print(net)
    cnt = 0
    for param in net.parameters():
        cnt += len(param.flatten())
    print(cnt)
    print(y.shape)
    assert tuple(y.shape) == (5,2,320,320)
    x = torch.randn(5, 4, 320, 320).cuda()
    y = torch.randn(5, 4, 320, 320).cuda()
    encX = Encoder(4, (64, 64*2, 64*4)).cuda()
    encY = Encoder(4, (63, 63*2, 63*4)).cuda()
    dec = Decoder(2, (62, 62*2, 62*4), (64+63, (64+63)*2, (64+63)*4)).cuda()
    x, y = encX(x), encY(y)
    zipped = list(map(partial(torch.cat, dim=1), zip(x, y)))
    z = dec(zipped)
    print(z.shape)
    enc1 = Encoder(4, (64, 64*2, 64*4)).cuda()
    dec1 = Decoder(2, (64, 64*2, 64*4), (64, 64*2, 64*4)).cuda()
    cnt = 0
    for param in enc1.parameters():
        cnt += len(param.flatten())
    for param in dec1.parameters():
        cnt += len(param.flatten())
    print(cnt)

