import math
import numpy as np
import torch
import torch.nn.functional as F
from miloss import gaussian_smooth

def lncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims ==  2, "volumes should be 2 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    #I2 = I*I
    #J2 = J*J
    #IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to(I)

    pad_no = math.floor(win[0]/2)

    stride = (1,1)
    padding = (pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def ms_lncc_loss(I, J, win=None, ms=3, sigma=3):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, sigma), kernel_size = 2, stride=2)
    loss = lncc_loss(I, J, win)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + lncc_loss(I, J, win)
    return loss / ms


