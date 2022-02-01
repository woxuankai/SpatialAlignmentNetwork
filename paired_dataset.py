#!/usr/bin/env python3

import os, sys, json
import numpy as np
import h5py
import torch

def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data

class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume, crop=None, q=0, flatten_channels=False):
        super().__init__()
        assert q < 0.5
        self.volume = volume
        self.flatten_channels = flatten_channels
        self.crop = crop
        h5 = h5py.File(volume, 'r')
        if len(h5['image'].shape) == 3:
            assert flatten_channels==False
            length, self.channels = h5['image'].shape[0], 1
        elif len(h5['image'].shape) == 4:
            length, self.channels = h5['image'].shape[0:2]
        else:
            assert False
        self.protocal = h5.attrs['acquisition']
        h5.close()
        self.start = round(length * q) # inclusive
        self.stop = length - self.start # exclusive

    def __len__(self):
        length = self.stop - self.start
        return length*self.channels if self.flatten_channels else length

    def __getitem__(self, index):
        h5 = h5py.File(self.volume, 'r')
        image = h5['image']
        if self.flatten_channels:
            i = h5['image'][index//self.channels + self.start]
            i = i[index%self.channels][()][None, ...]
        else:
            i = h5['image'][index + self.start][()]
            # extend channel for single-coiled data
            i = i if len(i.shape) == 3 else i[None, ...]
        #minVal = h5.attrs['minVal']
        minVal = 0
        maxVal = h5.attrs['max']
        h5.close()
        i = (i - minVal) / (maxVal - minVal)
        if self.crop is not None: i = center_crop(i, (self.crop, self.crop))
        # add dim for channel
        if len(i.shape) == 2: i = i[None, :, :]
        return i.astype(np.complex64)

class DummyVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, ref):
        super().__init__()
        sample = ref[0]
        self.shape = sample.shape
        self.dtype = sample.dtype
        self.len = len(ref)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return np.zeros(self.shape, dtype=self.dtype)

class AlignedVolumesDataset(torch.utils.data.Dataset):
    def __init__(self, *volumes, protocals, \
            crop=None, q=0, flatten_channels=False):
        super().__init__()
        volumes = [VolumeDataset(x, \
                crop, q=q, flatten_channels=flatten_channels) for x in volumes]
        assert len({len(x) for x in volumes}) == 1
        assert len({x[0].shape for x in volumes}) == 1
        self.crop = crop
        volumes = {volume.protocal:volume for volume in volumes}
        volumes['None'] = DummyVolumeDataset(next(iter(volumes.values())))
        for x in protocals:
            assert x in volumes.keys(), x+' not found in '+str(volumes.keys())
        volumes = [volumes[protocal] for protocal in protocals]
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes[0])

    def __getitem__(self, index):
        images = [volume[index] for volume in self.volumes]
        return images

def get_paired_volume_datasets(csv_path, protocals=None, crop=None, q=0, flatten_channels=False):
    datasets = []
    for line in open(csv_path, 'r').readlines():
        basepath = os.path.dirname(os.path.abspath(csv_path))
        dataset = [os.path.join(basepath, filepath) \
                for filepath in line.strip().split(',')]
        dataset = AlignedVolumesDataset(*dataset, \
                protocals=protocals, crop=crop, q=q, \
                flatten_channels=flatten_channels)
        datasets.append(dataset)
    return datasets #torch.utils.data.ConcatDataset(datasets)

class tiffPaired(torch.utils.data.Dataset):
        def __init__(self, tiffs, crop = None):
            super().__init__()
            self.tiffs = tiffs
            self.crop = crop
        
        def __len__(self):
            return len(self.tiffs)

        def __getitem__(self, ind):
            img = imageio.imread(self.tiffs[ind])
            assert len(img.shape) == 2
            t1, t2 = np.split(img, 2, axis=-1)
            t1, t2 = map(lambda x: np.stack([x, np.zeros_like(x)], axis=0), \
                    (t1, t2))
            if self.crop is not None:
                t1, t2 = map(lambda x: center_crop(x, [self.crop]*2), \
                        (t1, t2))
            return t1, t2

if __name__ == '__main__':
    # test get_paired_volume_datasets
    datasets = get_paired_volume_datasets( \
            '/mnt/btrfs/workspace/UII_paired/T1Flair_T2Flair_T2_train.csv', \
            protocals=['T1Flair', 'T2Flair', 'T2'],
            flatten_channels=True)
    print(sum([len(dataset) for dataset in datasets]))
    datasets = get_paired_volume_datasets( \
            '/mnt/btrfs/workspace/UII_paired/T1Flair_T2Flair_T2_train.csv', \
            protocals=['T1Flair', 'T2Flair', 'T2'])
    print(sum([len(dataset) for dataset in datasets]))
    #print(datasets[20][0].max())
    #print(datasets[20][1].max())
    #print(datasets[20][2].max())
    #print(datasets[20][3].max())
