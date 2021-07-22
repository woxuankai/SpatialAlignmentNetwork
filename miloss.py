import math
import numpy as np
import torch
import torch.nn.functional as F

def gaussian_kernel_1d(sigma):
    kernel_size = int(2*math.ceil(sigma*2) + 1)
    x = torch.linspace(-(kernel_size-1)//2, (kernel_size-1)//2, kernel_size)
    kernel = 1.0/(sigma*math.sqrt(2*math.pi))*torch.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel

def gaussian_kernel_2d(sigma):
    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])
    kernel = torch.tensordot(y_1, y_2, 0)
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(img, sigma):
    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(img)
    padding = kernel.shape[-1]//2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img

def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2*sigma**2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p



def _mi_loss(I, J, bins, sigma):
    # compute marjinal entropy
    ent_I, p_I = compute_marginal_entropy(I.view(-1), bins, sigma)
    ent_J, p_J = compute_marginal_entropy(J.view(-1), bins, sigma)
    # compute joint entropy
    normalizer_2d = 2.0 * np.pi*sigma**2
    p_joint = torch.mm(p_I, p_J.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
    ent_joint = -(p_joint * torch.log(p_joint + 1e-10)).sum()

    return -(ent_I + ent_J - ent_joint)


def mi_loss(I, J, bins=64 ,sigma=1.0/64, minVal=0, maxVal=1):
    #if sigma > 1:
    #    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(I)
    #    padding = kernel.shape[-1]//2
    #    I = torch.nn.functional.conv2d(I, kernel, padding=padding)
    #    J = torch.nn.functional.conv2d(J, kernel, padding=padding)
    bins = torch.linspace(minVal, maxVal, bins).to(I).unsqueeze(1)
    neg_mi =[_mi_loss(I, J, bins, sigma) for I, J in zip(I, J)]
    return sum(neg_mi)/len(neg_mi)

def ms_mi_loss(I, J, bins=64, sigma=1.0/64, ms=3, smooth=3, minVal=0, maxVal=1):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, smooth), kernel_size = 2, stride=2)
    loss = mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + mi_loss(I, J, \
                bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    return loss / ms

'''
def compute_joint_prob(x, y, bins):
    p = torch.exp(-(x-bins).pow(2).unsqueeze(1)-(y-bins).pow(2).unsqueeze(0))
    p_n = p.mean(dim=2)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return p_n

def _mi_loss1(I, J, bins):
    # in fact _mi_loss1 and _mi_loss works in the same way,
    # with minor different cased by numerical error
    Pxy = compute_joint_prob(I.view(-1), J.view(-1), bins)
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    PxPy = Px[..., None]*Py[None, ...]
    mi = Pxy*(torch.log(Pxy+1e-10) - torch.log(PxPy+1e-10))
    return -mi.sum()
'''

if __name__ == '__main__':
    #data = np.random.multivariate_normal( \
    #        mean=[0,0], cov=[[1,0.8],[0.8,1]], size=1000)
    #x, y = data[:,0], data[:,1]
    noise = 0.1
    x = np.random.random(512*512)*(1-noise)
    y = x + np.random.random(512*512)*noise


    from sklearn.metrics import mutual_info_score
    def calc_MI(x, y, bins):
        c_xy = np.histogram2d(x, y, bins, range=((0,1),(0,1)))[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi
    print(calc_MI(x, y, 64))

    #from scipy.stats import chi2_contingency
    #def calc_MI(x, y, bins):
    #    c_xy = np.histogram2d(x, y, bins)[0]
    #    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    #    mi = 0.5 * g / c_xy.sum()
    #    return mi
    #print(calc_MI(x, y, 60))

    from scipy.special import xlogy
    def calc_MI(x, y, bins):
        Pxy = np.histogram2d(x, y, bins, range=((0,1),(0,1)))[0]
        Pxy = Pxy/Pxy.sum()
        Px = Pxy.sum(axis=1)
        Py = Pxy.sum(axis=0)
        PxPy = Px[..., None]*Py[None, ...]
        #mi = Pxy * np.log(Pxy/(PxPy+1e-6))
        mi = xlogy(Pxy, Pxy) - xlogy(Pxy, PxPy)
        return mi.sum()
    print(calc_MI(x, y, 64))

    print(-mi_loss(torch.Tensor([x]), torch.Tensor([y]), bins=64 ,sigma=3, minVal=0, maxVal=1))
