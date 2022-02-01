import torch
import torch.fft

def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
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
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    #if torch.is_complex(x):
    #    return (x.real**2 + x.imag**2).sum(dim=1, keepdim=True).sqrt()
    #else:
    #    return (x**2).sum(dim=1, keepdim=True)**0.5

