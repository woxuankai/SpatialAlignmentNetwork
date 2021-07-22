#!/usr/bin/env python3

import os, sys, os.path, random, time, statistics, glob, math, json

import torch
import numpy as np
import tqdm
import nibabel as nib

from paired_dataset import get_paired_volume_datasets, center_crop
from basemodel import Config
from model import CSModel
from augment import augment


def main(args):
    Model = CSModel

    print(args)

    device = torch.device('cuda')
    if os.path.isfile(args.resume):
        net = Model(ckpt=args.resume)
        print('load ckpt from:', args.resume)
    else:
        raise FileNotFoundError
    cfg = net.cfg

    if args.aux_aug != 'None':
        volumes = get_paired_volume_datasets( \
                args.val, crop=int(cfg.shape*1.1), protocals=args.protocals)
    else:
        volumes = get_paired_volume_datasets( \
                args.val, crop=cfg.shape, protocals=args.protocals)
    net = net.to(device)
    net.eval()

    stat_eval  = []
    for i, volume in enumerate(volumes):
        batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in \
                zip(*[volume[j] for j in range(len(volume))])]
        with torch.no_grad():
            if args.aux_aug != 'None':
                batch =  [augment(x, bspline=(args.aux_aug=='BSpline'))[0] \
                        for x in batch]
                batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]
        net.set_input(*batch)
        with torch.no_grad():
            net.test()
            vis = net.get_vis('scalars')
            stat_eval.append(vis['scalars'])
        if args.save is None:
            continue
        image, sampled, aux, warped, rec, grid = \
                net.img_full_rss, net.img_sampled_rss, net.img_aux_rss, net.img_warped_rss, net.img_rec, net.net_T.offset
        grid = torch.stack([grid[...,0], grid[..., 1], torch.zeros_like(grid[...,0])], dim=-1)*(cfg.shape-1)/2
        grid = grid.permute(3, 0, 1, 2)[:,None,...]
        grid = nib.Nifti1Image(grid.cpu().numpy().T, np.eye(4))
        image, sampled, aux, warped, rec = [nib.Nifti1Image(x.cpu().squeeze(1).numpy().T, np.eye(4)) for x in (image, sampled, aux, warped, rec)]
        nib.save(image, args.save+'/'+str(i)+'_image.nii')
        nib.save(aux, args.save+'/'+str(i)+'_aux.nii')
        nib.save(sampled, args.save+'/'+str(i)+'_sampled.nii')
        nib.save(warped, args.save+'/'+str(i)+'_warped.nii')
        nib.save(rec, args.save+'/'+str(i)+'_rec.nii')
        nib.save(grid, args.save+'/'+str(i)+'_grid.nii')
    if args.metric is not None:
        with open(args.metric, 'w') as f:
            json.dump(stat_eval, f)
    vis = {key: statistics.mean([x[key] for x in stat_eval]) \
            for key in stat_eval[0]}
    print(vis)

if __name__ == '__main__':
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unknown boolean value.')

    parser = argparse.ArgumentParser(description='CS with adaptive mask')
    parser.add_argument('--resume', type=str, default=None, \
            help='with ckpt path, set empty str to load latest ckpt')
    parser.add_argument('--save', metavar='/path/to/save', \
            required=False, type=str, help='path to save evaluated data')
    parser.add_argument('--metric', metavar='/path/to/metric', \
            required=False, type=str, help='path to save metrics')
    parser.add_argument('--val', metavar='/path/to/evaluation_data', \
            required=True, type=str, help='path to evaluation data')
    parser.add_argument('--crop', type=int, default=320, \
            help='mask and image shape, images will be cropped to match')
    parser.add_argument('--protocals', metavar='NAME', \
            type=str, default=None, nargs='*',
            help='input modalities')
    parser.add_argument('--aux_aug', type=str, required=True, \
            choices=['None', 'Rigid', 'BSpline'],
            help='data augmentation aux image')
    args = parser.parse_args()

    main(args)

