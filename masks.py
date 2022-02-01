import math
import functools
import random
import torch
import numpy as np

class Mask(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_parameter('weight', None) # weights
        #self.weight = torch.nn.Parameter(torch.rand(shape))
        self.weight = torch.nn.Parameter(torch.ones(shape))
        # if a weight is already pruned, set to True
        self.register_buffer('pruned', None)
        self.pruned = torch.zeros(shape, dtype=torch.bool)

    def prune(self, num, thres=1, random=0):
        '''
        if not random, for abs(w) < thres,
            prune at most 'num' w_i from low to high, with abs(w)
        if random, for abs(w) < uniform[0-thres]
            prune at most 'num' w_i from low to high,
            with abs(w) - uniform[0-thres].
        '''
        assert thres >= 0 and random >= 0 and num >= 0
        if num == 0:
            return
        with torch.no_grad():
            w = self.weight.detach().abs()
            w.masked_scatter_(self.pruned, \
                    torch.ones_like(w)*(max(random, w.max())+thres))
            w.masked_scatter_(w >= thres, \
                    torch.ones_like(w)*(max(random, w.max())+thres))
            rand = torch.rand_like(w)*random
            _, ind = torch.topk(-(w-rand), num)
            ind = torch.masked_select(ind, w[ind] < thres)
            self.pruned.scatter_( \
                -1, ind, torch.ones_like(self.pruned))

    def forward(self, image):
        mask = torch.ones_like(self.weight)
        mask.masked_scatter_(self.pruned, torch.zeros_like(self.weight))
        # unable to set a leaf variable here
        #self.weight.masked_scatter_(self.pruned, torch.zeros_like(self.weight))
        # mask weight in preventation of weight changing
        return image * (self.weight*mask)[None, None, None, :]

class StandardMask(Mask):
    """When  the  acceleration factorequals four,
    the fully-sampled central region includes 8% of all k-space lines;
    when it equals eight, 4% of all k-space lines are included.
    """
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        center_ratio = sparsity*0.32 # i.e. 4% for 8-fold and 8% for 4-fold
        center_len = round(shape * center_ratio) # to round up to int
        other_ratio = (sparsity*shape - center_len)/(shape - center_len)
        prob = torch.ones(shape)*1.1
        # low freq is of the border
        prob[center_len//2:center_len//2-center_len] = other_ratio
        thresh = torch.rand(shape)
        _, ind = torch.topk(prob - thresh, math.floor(sparsity*shape), dim=-1)
        self.pruned = \
                torch.ones_like(thresh, dtype=torch.bool).scatter( \
                -1, ind, torch.zeros_like(thresh, dtype=torch.bool))


'''
class RandomMask(Mask):
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        """
        super().__init__(shape)
        _, ind = torch.topk(torch.rand(shape), math.floor(sparsity*shape), dim=-1)
        self.pruned = \
                torch.ones(shape, dtype=torch.bool).scatter( \
                -1, ind, torch.zeros(shape, dtype=torch.bool))
'''

class EquispacedMask(Mask):
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        center_ratio = sparsity*0.32 # i.e. 4% for 8-fold and 8% for 4-fold
        center_len = round(shape * center_ratio) # to round up to int
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        # low freq is of the border
        self.pruned[center_len//2:center_len//2-center_len] = True
        remaining_cnt = math.floor(sparsity*shape - center_len)
        interval = int((shape-center_len-1)//(remaining_cnt-1))
        start_max = (shape - center_len) - \
                ((remaining_cnt-1)*interval + 1) # inclusive
        start = random.randint(0, start_max)
        pruned_part = \
                self.pruned[center_len//2:center_len//2-center_len].clone()
        pruned_part = torch.roll(pruned_part, pruned_part.shape[0]//2)
        #print(shape-center_len, remaining_cnt, interval, start_max, start, pruned_part.shape)
        pruned_part[start:start+interval*remaining_cnt:interval] = False
        pruned_part = torch.roll(pruned_part, (pruned_part.shape[0]+1)//2)
        # pytorch is buggy if just set False to uncloned pruned_part
        self.pruned[center_len//2:center_len//2-center_len] = pruned_part

class LowpassMask(Mask):
    """Low freq only
    """
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        """
        super().__init__(shape)
        #center_len = int(shape * sparsity+0.5) # to round up to int
        center_len = math.floor(shape * sparsity) # floor to int
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        # low freq is of the border
        self.pruned[center_len//2:center_len//2-center_len] = True

def rescale_prob(x, sparsity):
    """
    Rescale Probability x so that it obtains the desired sparsity
    if mean(x) > sparsity
      x' = x * sparsity / mean(x)
    else
      x' = 1 - (1-x) * (1-sparsity) / (1-mean(x))
    """
    xbar = x.mean()
    if xbar > sparsity:
        return x * sparsity / xbar
    else:
        return 1 - (1 - x) * (1 - sparsity) / (1 - xbar)

class LOUPEMask(torch.nn.Module):
    def __init__(self, sparsity, shape, pmask_slope=5, sample_slope=12):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        sample_slope: float, slope for soft threshold
        mask_param -> (sigmoid+rescale) -> pmask -> (sample) -> mask
        """
        super().__init__()
        assert sparsity <= 1 and sparsity >= 0
        self.sparsity = sparsity
        self.shape = shape
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.register_parameter('weight', None) # weights
        self.register_buffer('pruned', None)
        # eps could be very small, or somethinkg like eps = 1e-6
        # the idea is how far from the tails to have your initialization.
        eps = 0.01
        x = torch.rand(self.shape)*(1-eps*2) + eps
        # logit with slope factor
        self.weight = torch.nn.Parameter( \
                -torch.log(1. / x - 1.) / self.pmask_slope)
        self.forward(torch.randn(1, 1, shape, shape)) # to set self.mask

    def forward(self, example):
        assert example.shape[-1] == self.shape
        if False:
            mask = torch.zeros_like(self.weight)
            _, ind = torch.topk(self.weight, \
                    int(self.sparsity*self.shape+0.5), dim=-1)
            mask.scatter_(-1, ind, torch.ones_like(self.weight))
            self.pruned = (mask < 0.5)
            return example * mask[None, None, None, :]
        pmask = rescale_prob( \
                torch.sigmoid(self.weight*self.pmask_slope), \
                self.sparsity)
        thresh = torch.rand(example.shape[0], self.shape).to(pmask)
        _, ind = torch.topk(pmask - thresh, \
                int(self.sparsity*self.shape+0.5), dim=-1)
        not_pruned = torch.zeros_like(thresh).scatter( \
                -1, ind, torch.ones_like(thresh))
        self.pruned = (not_pruned < 0.5)[0]
        if self.training:
            mask = torch.sigmoid((pmask - thresh) * self.sample_slope)
            return example*mask[:, None, None, :]
        else:
            return example*(not_pruned)[:, None, None, :]

    def prune(self, num, thres=1, random=False):
        # nothing happened
        pass


class TaylorMask(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('weight', None)
        #self.weight = torch.ones(shape) # for compactibility
        #self.register_parameter('weight', None) # weights
        #self.weight = torch.nn.Parameter(torch.ones(shape))
        # if a weight is already pruned, set to True
        self.shape = shape
        self.register_buffer('pruned', None)
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        self.values = []


        '''
        def value_forward_hook(self, input, output):
            self.ouptup = output.detach()
        '''

    def prune(self, num, *args, **kwargs):
        #print('BOOM!!')
        w = self.values
        self.values = []
        if num == 0:
            # used to reset values only
            return
        assert num > 0 and len(w) > 0
        # exclude $num data with
        with torch.no_grad():
            w = torch.stack(w, 0).mean(0)
            w.masked_scatter_(self.pruned, torch.zeros_like(w))
            self.weight = w
            w.masked_scatter_(self.pruned, w.max()*torch.ones_like(w))
            _, ind = torch.topk(-w, num)
            self.pruned.scatter_(-1, ind, torch.ones_like(self.pruned))


    def forward(self, image):
        def value_backward_hook(self, grad):
            self.values.append(grad.detach()**2)
            #print('HA!!')

        wrapper = functools.partial(value_backward_hook, self)
        functools.update_wrapper(wrapper, value_backward_hook)

        self.mask = torch.ones(self.shape).to(image)
        self.mask.masked_scatter_(self.pruned, torch.zeros_like(self.mask))
        self.mask.requires_grad=True
        self.mask.register_hook(wrapper)
        return image * self.mask[None, None, None, :]

if __name__ == '__main__':
    sparsity = 1.0/8
    print(sparsity)
    shape = 256
    example = torch.rand(5, 2, shape*2, shape)
    mask = Mask(shape)
    print(mask.pruned.numpy().astype(np.float).mean())
    standard = StandardMask(sparsity, shape)
    print(standard.pruned.numpy().astype(np.float).mean())
    rand = RandomMask(sparsity, shape)
    print(rand.pruned.numpy().astype(np.float).mean())
    lowpass = LowpassMask(sparsity, shape)
    print(lowpass.pruned.numpy().astype(np.float).mean())
    equispaced = EquispacedMask(sparsity, shape)
    print(equispaced.pruned.numpy().astype(np.float).mean())
    loupe = LOUPEMask(sparsity, shape)
    print(loupe.pruned.numpy().astype(np.float).mean())

