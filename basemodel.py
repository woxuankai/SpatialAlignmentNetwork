import os
import os.path
import numpy as np
import glob
import torch

'''
Ideas on deduplication between checkpoints:
    1. Substitue the f in torch.save with a io.ByteIO file.
        After torch.save, extract the zipfile to destination.
    2. Save objects seperately in the same folder with torch.save.
    3. More fine-grained saving: save each key:val in state_dict seperately.
'''
# class FolderAsZipIO(object):
#     def __init__(self, folder=None):
#         super(self).__init__()
#         self.fd = io.ByteIO()
# 
# def save_ckpt(f, state_dict):
#     assert not os.path.exists(f), folder+' already exists'
#     #os.mkdir(folder)
#     state_dict = {key:val.cpu().numpy() for key, val in state_dict.items()}
#     np.savez(os.path.join(f, '.npy'), **state_dict)
#     # for key, val in state_dict.items():
#     #     # with open(folder+'/'+key, 'wb') as f:
#     #     #    f.write(
#     #     np.save(os.path.join(folder, key, 'npy'),
#     return None
# 
# def load_state_dicts(f):
#     assert os.path.isdir(f), folder+' is not a directory'


def ckpt_load(folder):
    assert os.path.isdir(folder)
    ckpt = {f:torch.load(os.path.join(folder, f), map_location='cpu') \
            for f in os.listdir(folder)}
    return ckpt

def ckpt_save(ckpt, folder):
    assert isinstance(ckpt, dict)
    assert not os.path.exists(folder)
    os.mkdir(folder)
    for key, val in ckpt.item():
        torch.save(val, os.path.join(folder, key))

class Config(object):
    def __init__(self, **params):
        super().__init__()
        super().__setattr__('memo', [])
        for key, val in params.items():
            setattr(self, key, val)

    def __setattr__(self, name, value):
        self.memo.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self.memo.pop(self.memo.index(name))
        super().__delattr__(name)

    def __str__(self):
        return 'class Config containing: ' \
                + str({key: getattr(self, key) for key in self.memo})

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, param):
        assert param in self.memo, str(param)+' not found, try '+str(self.memo)
        return getattr(self, param)

    def __contains__(self, item):
        return item in self.memo

class BaseModel(object):
    def __init__(self, cfg=None, ckpt=None):
        super().__init__()
        if ckpt is not None:
            self.load(cfg=cfg, ckpt=ckpt)
        else:
            self.build(cfg=cfg)

    def build(self, config):
        self.cfg = config

    def to(self, device):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module) \
                    or isinstance(value, torch.Tensor):# \
                #    or isinstance(value, torch.optim.Optimizer):
                value.to(device)
            if isinstance(value, torch.optim.Optimizer):
                for param in value.state.values():
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                subparam.data = subparam.data.to(device)
                                if subparam._grad is not None:
                                    subparam._grad.data = \
                                            subparam._grad.data.to(device)
        return self

    def train(self, mode=True):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module):
                value.train(mode)
        self.training=mode
        return self

    def eval(self):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module):
                value.eval()
        self.training=False
        return self

    def get_saveable(self):
        return {key: value for key, value in self.__dict__.items() \
                if isinstance(value, torch.nn.Module)}# \
        #        or isinstance(value, torch.optim.Optimizer)}
        '''if hasattr(value, 'state_dict') \
                    and callable(value.state_dict) \
                    and hasattr(value, 'load_state_dict') \
                    and callable(value.load_state_dict):
        '''

    def save(self, ckpt, **objects):
        if len(objects) == 0:
            objects = self.get_saveable()
        objects = {key:value.state_dict() \
                for key, value in objects.items()}
        if hasattr(self, 'cfg'):
            objects['config'] = self.cfg
        else:
            print('!!! Missing cfg while saving !!!')
        torch.save(objects, ckpt)

    def load(self, ckpt, cfg=None, **objects):
        ckpt = torch.load(ckpt, map_location='cpu')
        if cfg is None:
            cfg = ckpt.pop('config')
        self.build(cfg=cfg)
        if len(objects) == 0:
            objects = self.get_saveable()
        for key, value in objects.items():
            if key in ckpt.keys():
                value.load_state_dict(ckpt[key])
            else:
                print('!!! Missing key '+key+' in checkpoint !!!')

if __name__ == '__main__':
    cfg = Config(var1=1, var2=2)
    cfg.var3 = 3
    del cfg.var2
    print(cfg)
    cfg.var1
    cfg['var1']
    model = BaseModel(cfg)
    model.save('/tmp/feel_free_to_delete_it.pth')
    model = BaseModel('/tmp/feel_free_to_delete_it.pth')
