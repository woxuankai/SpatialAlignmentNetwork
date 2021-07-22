#!/usr/bin/env python3

import os
import sys
import json
import torch
import numpy as np

def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def get_paired_volume_datasets(folder_path, csv_path, crop=None, q=0):
    volumes = get_volumes(folder_path)
    datasets = []
    for line in open(csv_path, 'r').readlines():
        pd, pdfs, _ = line.split(',')
        dataset = AlignedVolumesDataset( \
                volumes[pd], volumes[pdfs], crop=crop, q=q) 
        datasets.append(dataset)
    return datasets

class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume, crop=None):
        super().__init__()
        self.crop = crop
        nums = [key for key in volume if type(key)==int]
        #assert max(nums) == len(nums)-1, 'missing slices in '+volume[0]+'...'
        self.slices = [volume[i] for i in range(len(nums))]
        self.max_val = volume['max']

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        image = np.load(self.slices[index])
        #item = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(item)))
        #if self.crop is not None: image = centerCrop(image, self.crop)
        #item = item/(max_val*np.sqrt(item.shape[1]*item.shape[0]))
        image = image / self.max_val
        #item_abs = np.absolute(item)[..., np.newaxis]
        image = np.stack([np.real(image), np.imag(image)], 0)
        if self.crop is not None: image = center_crop(image, self.crop)
        return image.astype(np.float32)

class AlignedVolumesDataset(torch.utils.data.Dataset):
    def __init__(self, *volumes, crop=None, q=0):
        '''for each volume, must contain key 'slices' and 'max'
        '''
        super().__init__()
        volumes = [VolumeDataset(x) for x in volumes]
        assert len({len(x) for x in volumes}) == 1
        assert len({x[0].shape for x in volumes}) == 1
        assert q < 0.5
        self.volumes = volumes
        self.crop = crop
        num_slices = len(self.volumes)
        self.start = round(num_slices*q) # inclusive
        self.stop = num_slices - self.start # exclusive

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        images = [volume[self.start+index] for volume in self.volumes]
        if self.crop is not None:
            images = [center_crop(image, self.crop) for image in images]
        return images

def get_volumes(folder):
    '''collect slices from splitted fastMRI volumes
    '''
    assert os.path.isdir(folder)
    suffix = '.json'
    volumes = {}
    for attr in os.listdir(folder):
        if not attr.endswith(suffix): continue
        volumeID = attr[:-len(suffix)]
        with open(os.path.join(folder, attr), 'r') as f:
            volumes[volumeID] = json.load(f)
    for path in os.listdir(folder):
        suffix = '.npy'
        if ('-raw-' not in path) or (not path.endswith(suffix)): continue
        volumeID, _, num = path[:-len(suffix)].split('-')
        volumes[volumeID][int(num)] = os.path.join(folder, path)
    return volumes

def get_aligned_volumes(folder):
    '''select paired volumes only
    '''
    volumes = get_volumes(folder)
    patients = {}
    for volumeID in volumes:
        volume = volumes[volumeID]
        patient_id = volume['patient_id']
        if patient_id not in patients:
            patients[patient_id] = {}
        patients[patient_id][volume['acquisition']] = volume
    aligned_volumes = []
    for patient in patients.values():
        if ('CORPD_FBK' not in patient) or ('CORPDFS_FBK' not in patient):
            continue
        pd = patient['CORPD_FBK']
        pdfs = patient['CORPDFS_FBK']
        aligned_volumes.append((pd, pdfs))
    return aligned_volumes

if __name__ == '__main__':

    def MI(x, y, bins=200):
        eps = 1e-6
        x, y = map(lambda x: np.clip(x, 0, 1).ravel().astype(np.float64), (x, y))
        pxy, _, _ = np.histogram2d( \
                x, y, bins, range=((0, 1), (0, 1)))
        pxy = pxy*1.0/pxy.sum()
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, np.newaxis] * py[np.newaxis, :]
        return np.sum(pxy*np.log((pxy+eps)/(px_py+eps)))
    
    # test get_volume
    volumes = get_volumes(sys.argv[1])
    print(len(volumes))
    # test VolumeDataset
    dataset = [VolumeDataset(x) for x in volumes.values()]
    # test get_aligned_volumes
    volume_pairs = get_aligned_volumes(sys.argv[1])
    print(len(volume_pairs))
    # test AlignedVolumesDataset
    import imageio
    cnt = 0
    MIs = []
    for volume_pd, volume_pdfs in volume_pairs:
        try:
            dataset = AlignedVolumesDataset(volume_pd, volume_pdfs)
        except Exception as e:
            continue
        pd, pdfs = map(lambda x: center_crop(np.array(x), (256, 256)), \
                zip(*[(x[0][0], x[1][0]) for x in dataset]))
        mi = MI(pd, pdfs)
        print(cnt, len(dataset), \
                volume_pd[0].split('/')[-1].split('-')[0], \
                volume_pdfs[0].split('/')[-1].split('-')[0], mi, \
                sep=',')
        if len(sys.argv) >= 3:
            for offset, (x, y) in enumerate(zip(pd, pdfs)):
                img = np.concatenate((x, np.ones((x.shape[0], 5)), y), 1)
                img = np.clip(np.floor(img*256),0,255).astype(np.uint8)
                imageio.imsave( \
                        sys.argv[2]+'/{:010d}'.format(cnt+offset)+'.jpg', img)
        MIs.append(mi)
        cnt += len(dataset)
    print(len(MIs), np.max(MIs), np.min(MIs), np.mean(MIs), np.std(MIs))
    import matplotlib.pyplot as plt
    plt.hist(MIs)
    plt.show()
    # test get_paired_volume_datasets
    datasets = get_paired_volume_datasets('/mnt/data/fastMRI/singlecoil_val_split/','/mnt/data/fastMRI/singlecoil_val_paried.csv')
