import os
import os.path
import glob
import json
import zipfile
import numpy as np
import torch

'''
Ideas on deduplication between checkpoints:
    1. Substitue the f in torch.save with a io.ByteIO file.
        After torch.save, extract the zipfile to destination.
   *2. Save objects seperately in the same folder with np.savez.
    3. More fine-grained saving: save each key:val in state_dict seperately.
'''

def ckpt_load(folder):
    if os.path.isfile(folder):
        return torch.load(folder)
    ckpt = {}
    for key in os.listdir(folder):
        save_path = os.path.join(folder, key)
        if key == 'config':
            try:
                ckpt[key] = Config()
                ckpt[key].load(save_path)
            except UnicodeDecodeError:
                ckpt[key] = torch.load(save_path)
        #elif zipfile.is_zipfile(save_path):
        #    ckpt[key] = torch.load(save_path, map_location='cpu')
        #else:
        #    ckpt[f] = np.load(save_path)
        #    ckpt[f] = {k:torch.from_numpy(v) for k, v in ckpt[f].items()}
        else:
            try:
                ckpt[key] = torch.load(save_path, map_location='cpu')
            except RuntimeError as e:
                ckpt[key] = np.load(save_path)
                ckpt[key] = {
                        k:torch.from_numpy(v) for k, v in ckpt[key].items()}
    return ckpt

def ckpt_save(ckpt, folder):
    assert isinstance(ckpt, dict)
    assert not os.path.exists(folder), folder+' already exists'
    os.mkdir(folder)
    for key, val in ckpt.items():
        save_path = os.path.join(folder, key)
        if key == 'config':
            val.save(save_path)
        else:
            val = {k:v.cpu().numpy() for k, v in val.items()}
            f = open(save_path, 'wb')
            np.savez(f, **val)
            f.close()

class Config(object):
    def __init__(self, **params):
        super().__init__()
        super().__setattr__('memo', [])
        for key, val in params.items():
            setattr(self, key, val)

    def __setattr__(self, name, value):
        if name not in self.memo:
            self.memo.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self.memo.remove(name)
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

    def load(self, save_path):
        for k in self.memo.copy():
            self.pop(k)
        f = open(save_path, 'r')
        content = json.load(f)
        f.close()
        for k, v in content.items():
            setattr(self, k, v)

    def save(self, save_path):
        content = {k:getattr(self, k) for k in self.memo}
        f = open(save_path, 'w')
        json.dump(content, f)
        f.close()

class BaseModel(object):
    def __init__(self, cfg=None, ckpt=None, objects=None):
        super().__init__()
        if ckpt is not None:
            self.load(cfg=cfg, ckpt=ckpt, objects=objects)
        else:
            self.build(cfg=cfg)
        self.training = True

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

    def save(self, ckpt, objects=None):
        saveable = self.get_saveable()
        if objects is None:
            objects = saveable.keys()
        objects = {key: saveable[key].state_dict() for key in objects}
        if hasattr(self, 'cfg'):
            objects['config'] = self.cfg
        else:
            print('!!! Missing cfg while saving !!!')
        #torch.save(objects, ckpt)
        ckpt_save(objects, ckpt)

    def load(self, ckpt, cfg=None, objects=None):
        #ckpt = torch.load(ckpt, map_location='cpu')
        ckpt = ckpt_load(ckpt)
        if cfg is None:
            cfg = ckpt.pop('config')
        self.build(cfg=cfg)
        saveable = self.get_saveable()
        if objects is None:
            objects = saveable.keys()
        objects = {key: saveable[key] for key in objects}
        for key, value in objects.items():
            value.load_state_dict(ckpt[key])

if __name__ == '__main__':
    import sys, os
    import shutil
    ckpt = ckpt_load(sys.argv[1])
    if len(sys.argv) >= 3:
        ckpt_save(ckpt, sys.argv[2])
    else:
        if os.path.isdir(sys.argv[1]):
            shutil.rmtree(sys.argv[1])
        elif os.path.isfile(sys.argv[1]):
            os.remove(sys.argv[1])
        else:
            assert False
        ckpt_save(ckpt, sys.argv[1])

    '''
    cfg = Config(var1=1, var2=2)
    cfg.var3 = 3
    del cfg.var2
    print(cfg)
    cfg.var1
    cfg['var1']
    model = BaseModel(cfg)
    model.save('/tmp/feel_free_to_delete_it.pth')
    model = BaseModel('/tmp/feel_free_to_delete_it.pth')
    '''
