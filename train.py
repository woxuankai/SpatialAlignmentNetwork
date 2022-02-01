#!/usr/bin/env python3

import os, sys, os.path
import random, time, statistics, glob, math
import shutil
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
# set random seed just for more consistent visualization
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import torch.utils.tensorboard
import torchvision
import torchvision.utils
import tqdm

from paired_dataset import get_paired_volume_datasets, center_crop
from basemodel import Config
from model import CSModel
from augment import augment

class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm.tqdm(dataset, leave=False)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]

def augment_None(batch):
    return batch

def augment_Rigid(batch):
    return [augment(x, rigid=True, bspline=False)[0] for x in batch]

def augment_BSpline(batch):
    return [augment(x, rigid=True, bspline=True)[0] for x in batch]

def augment_PBSpline(batch):
    returnVal = []
    grid = None
    for x in batch:
        if grid is None:
            x, grid = augment(x, rigid=True, bspline=True)
        else:
            x, _ = augment(x, rigid=False, bspline=False, grid=grid)
        returnVal.append(x)
    return returnVal

augment_funcs = { \
        'None': augment_None,
        'Rigid': augment_Rigid,
        'BSpline': augment_BSpline,
        'PBSpline': augment_PBSpline}

def main(args):
    # setup
    cfg = Config()
    cfg.sparsity = args.sparsity
    cfg.lr = args.lr
    #cfg.mask_lr = args.mask_lr
    cfg.shape = args.crop
    cfg.coils = args.coils
    #cfg.tt = args.tt
    cfg.reg = args.reg
    #cfg.rec = args.rec
    cfg.mask = args.mask
    cfg.weight_smooth = args.smooth_weight
    cfg.weight_gan = args.gan_weight
    cfg.weight_gan_sim = args.gan_sim_weight
    cfg.weight_sim = args.sim_weight
    cfg.use_amp = args.use_amp
    #cfg.sim = args.sim_weight
    #cfg.mask_losses = {}
    #for item in args.mask_losses:
    #    loss_name, loss_weight = item.split(':')
    #    loss_weight = float(loss_weight)
    #    cfg.mask_losses[loss_name] = loss_weight
    #cfg.rec_losses = {}
    #for item in args.rec_losses:
    #    loss_name, loss_weight = item.split(':')
    #    loss_weight = float(loss_weight)
    #    cfg.rec_losses[loss_name] = loss_weight
    Model = CSModel

    print(args)
    for path in [args.logdir, args.logdir+'/res', args.logdir+'/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    print('loading model...')
    seed = 19950102+666+233
    random.seed(seed)
    device = torch.device('cuda')
    iter_cnt = 0
    ckpt = None
    if args.resume is not None:
        if args.resume == '': # load latest
            ckpts = glob.glob(args.logdir+'/ckpt/ckpt_*.pt')
            ckpts += glob.glob(args.logdir+'/ckpt/ckpt_*.pth')
            if len(ckpts) == 0:
                print('no avaliable ckpt found.')
                raise FileNotFoundError
            ckpts = sorted(ckpts, key=os.path.getmtime)
            ckpt = ckpts[-1]
            iter_cnt = int(ckpt.split('.')[-2].split('_')[-1])
            print('Will load latest ckpt from:', ckpt,
                    ', cnt:', iter_cnt,
                    ', load nets:', args.load_nets)
        else: # load specific ckpt
            print('Will load specified ckpt from:', args.resume,
                    ', cnt:', iter_cnt,
                    ', load nets:', args.load_nets)
            ckpt = args.resume
        net = Model(ckpt=ckpt, cfg=cfg, objects=args.load_nets)
    else:
        assert args.load_nets is None
        print('training from scratch...')
        net = Model(cfg=cfg)

    print(net.cfg)
    cfg = net.cfg
    random.seed(int(time.time()))

    writer.add_text('date', repr(time.ctime()))
    writer.add_text('working dir', repr(os.getcwd()))
    writer.add_text('__file__', repr(os.path.abspath(__file__)))
    writer.add_text('commands', repr(sys.argv))
    writer.add_text('arguments', repr(args))
    writer.add_text('actual config', repr(cfg))
    writer.add_text('ckpt', repr(ckpt))

    print('loading data...')
    volumes_train = get_paired_volume_datasets( \
            args.train, crop=int(cfg.shape*1.1), protocals=args.protocals)
    #        flatten_channels=True)
    #        args.train, crop=cfg.shape, q=1/5.)
    volumes_val = get_paired_volume_datasets( \
            args.val, crop=cfg.shape, protocals=args.protocals)
    #        flatten_channels=True)
    #        args.val, crop=cfg.shape, q=1/5.)
    slices_train = torch.utils.data.ConcatDataset(volumes_train)
    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    if args.prefetch:
        # load all data to ram
        slices_train = Prefetch(slices_train)
        slices_val = Prefetch(slices_val)
    loader_train = torch.utils.data.DataLoader( \
            slices_train, batch_size=args.batch_size, shuffle=True, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = torch.utils.data.DataLoader( \
            slices_val, batch_size=args.batch_size, shuffle=False, \
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    len_vis = 16
    col_vis = 4
    seed = 19950102+666+233
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_vis = next(iter(torch.utils.data.DataLoader( \
            slices_val, batch_size=len_vis, shuffle=True)))
    batch_vis = [x.to(device, non_blocking=True) for x in batch_vis]
    #if args.aux_aug != 'None':
    #    batch_vis[0], _ = augment( \
    #            batch_vis[0], bspline=(args.aux_aug=='BSpline'))
    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))
    print('done, ' \
            + str(len(slices_train)) + ' / ' \
            + str(len(volumes_train)) + ' for training, ' \
            + str(len(slices_val)) + ' / ' \
            + str(len(volumes_val)) + ' for validation')

    # training.
    print('training...')
    net = net.to(device)
    last_loss, last_ckpt, last_disp = 0, 0, 0
    time_data, time_vis = 0, 0
    signal_end = False
    iter_best = iter_cnt
    loss_best = None

    time_start = time.time()
    #t0 = time.time()
    for num_epoch in tqdm.trange(args.epoch, desc='epoch', leave=True):
        ###################  training ########################
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter', \
                bar_format=str(args.batch_size)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        if signal_end:
            break
        for batch in tqdm_iter:
        #for batch in loader_train:
            if signal_end:
                break
            net.train()
            time_data = time.time() - time_start

            #t1 = time.time()
            iter_cnt += 1
            batch = [x.to(device, non_blocking=True) for x in batch]
            with torch.no_grad():
                batch = augment_funcs[args.aux_aug](batch)
                batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]
            #t2 = time.time()
            net.set_input(*batch)
            #t3 = time.time()
            #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
            #    net.update()
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            net.update()
            del batch
            #t4 = time.time()

            time_start = time.time()
            if iter_cnt % 50 == 0:
                last_loss = iter_cnt
                vis = net.get_vis('scalars')
                for name, val in vis['scalars'].items():
                    writer.add_scalar('train/'+name, val, iter_cnt)
                vis = net.get_vis('histograms')
                for name, val in vis['histograms'].items():
                    writer.add_histogram( \
                            tag='train/'+name, \
                            global_step=iter_cnt, \
                            **val)
                del vis, name, val
            if (iter_cnt % 1000 == 0) or \
                    ((iter_cnt < 10000) and (iter_cnt % 100 == 0)):
                last_disp = iter_cnt
                net.eval()
                # visualize image
                with torch.no_grad():
                    net.set_input(*batch_vis)
                    net.test()
                vis = net.get_vis('images')
                for name, val in vis['images'].items():
                    torchvision.utils.save_image(val, \
                        args.logdir+'/res/'+'%010d_'%iter_cnt+name+'.jpg', \
                        nrow=len_vis//col_vis, padding=10, \
                        range=(0, 1), pad_value=0.5)
                del vis, name, val
            if (iter_cnt % 5000 == 0) or \
                    ((iter_cnt < 10000) and (iter_cnt % 1000 == 0)):
                # 3000 should be dividable by 250
                last_ckpt = iter_cnt
                net.save(args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)
            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)
            #t5 = time.time()
            #print(t5-t0, t5-t4, t4-t3, t3-t2, t2-t1, t1-t0)
            #t0 = time.time()

        ###################  validation  ########################
        net.eval()
        tqdm_iter = tqdm.tqdm(loader_val, desc='iter', \
                bar_format=str(args.batch_size)+'(val) {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        stat_eval  = []
        stat_loss = []
        time_start = time.time()
        with torch.no_grad():
            for batch in tqdm_iter:
                time_data = time.time() - time_start
                batch = [x.to(device, non_blocking=True) for x in batch]
                net.set_input(*batch)
                stat_loss.append(net.test())
                del batch
                vis = net.get_vis('scalars')
                stat_eval.append(vis['scalars'])
                time_start = time.time()
                if time_data >= 0.1:
                    postfix += ' data %.1f'%time_data
            vis = {key: statistics.mean([x[key] for x in stat_eval]) \
                    for key in stat_eval[0]}
            for name, val in vis.items():
                writer.add_scalar('val/'+name, val, iter_cnt)
            loss_current = statistics.mean(stat_loss)
            del vis
            if args.intel_stop > 0:
                # intel_stop is enabled
                if (loss_best is None) or (loss_current < loss_best):
                    # new record
                    loss_best = loss_current
                    iter_best = iter_cnt
                    if os.path.exists(args.logdir+'/ckpt/best.pt'):
                        shutil.rmtree(args.logdir+'/ckpt/best.pt')
                    net.save(args.logdir+'/ckpt/best.pt')
                else:
                    # worse than best
                    if iter_cnt >= args.intel_stop + iter_best:
                        # no better result after intel_stop iterations
                        signal_end=True
                        print('signal_end set due to intel_stop')
            #writer.add_scalar('val/temp_loss_best', loss_best, iter_cnt)
        #writer.add_scalar('val/temp_loss_current', loss_current, iter_cnt)

    print('reached end of training loop, and signal_end is '+str(signal_end))
    writer.flush()
    writer.close()
    net.save(args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)
    print('saved final ckpt:', args.logdir+'/ckpt/ckpt_%010d.pt'%iter_cnt)


if __name__ == '__main__':
    import argparse
    from autoGPU import autoGPU

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unknown boolean value.')

    def try_int(v):
        # convert string to int
        try:
            v = int(v)
        except ValueError:
            v = int(float(v))
        assert v >= 0
        return v

    parser = argparse.ArgumentParser(description='CS with adaptive mask')
    parser.add_argument('--logdir', metavar='logdir', \
            type=str, required=True, help='path for storage and checkpoint')
    parser.add_argument('--resume', type=str, default=None, \
            help='with ckpt path, set empty str to load latest ckpt')
    parser.add_argument(
            '--load_nets',
            type=str,
            nargs='*',
            default=None,
            help='neural networks to be loaded in the checkpoint')
    parser.add_argument('--epoch', type=int, default=150, \
            help='epochs to train')
    parser.add_argument('--batch_size', type=int, default=10, \
            help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), \
            help='number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=1e-4, \
            help='learning rate')
    parser.add_argument('--intel_stop', type=try_int, default=0, metavar='N', \
            help='stop training after val loss not going down for N iters')
    parser.add_argument('--reg', metavar='registration singal loss', \
            type=str, required=True, \
            choices=['None', 'Rec', 'Mixed', 'GAN-Only'],\
            help='[None (Reconstruction Only), Rec, Mixed, GAN-Only]')
    #parser.add_argument('--rec', metavar='run registration', \
    #        type=str2bool, required=True, help='[True, False]')
    #parser.add_argument('--tt', type=int, nargs='*', \
    #        help='layers to use texture transformation (tarting from 0)', \
    #        metavar='e.g. 2 3')
    # losses
    #parser.add_argument('--rec_losses', type=str, required=True, nargs='*', \
    #        help='losses for reconstruction', \
    #        metavar='NAME1:WEIGHT1 NAME2:WEIGHT2')
    #parser.add_argument('--mask_losses', type=str, required=True, nargs='*', \
    #        help='losses for mask',
    #        metavar='NAME1:WEIGHT1 NAME2:WEIGHT2')
    parser.add_argument('--smooth_weight', type=float, required=True, \
            help='weight for deformation field smoothness',
            metavar='Float')
    parser.add_argument('--gan_weight', type=float, required=True, \
            help='weight for discriminator',
            metavar='Float')
    parser.add_argument('--gan_sim_weight', type=float, required=True, \
            help='weight for cross modality synthesis',
            metavar='Float')
    parser.add_argument('--sim_weight', type=float, required=True, \
            help='weight for reconstruction similarity loss',
            metavar='Float')
    # mask
    parser.add_argument('--mask', metavar='type', \
            #choices=['learnable', 'uniform', 'standard'], \
            required=True, type=str, help='types of mask')
    parser.add_argument('--sparsity', metavar='0-1', \
                type=float, default=None, \
                help='desired overall sparisity of masks without sparsity')
    #parser.add_argument('--mask_lr', type=float, default=1e-3, \
    #        help='learning rate for mask')
    # data
    parser.add_argument('--train', metavar='/path/to/training_data', \
            required=True, type=str, help='path to training data')
    parser.add_argument('--val', metavar='/path/to/validation_data', \
            required=True, type=str, help='path to validation data')
    parser.add_argument('--crop', type=int, default=320, \
            help='mask and image shape, images will be cropped to match')
    parser.add_argument('--coils', type=int, default=1, \
            help='number of coils')
    parser.add_argument('--protocals', metavar='NAME', \
            type=str, default=None, nargs='*',
            help='input modalities')
    parser.add_argument('--aux_aug', type=str, required=True, \
            choices=augment_funcs.keys(),
            help='data augmentation aux image')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--force_gpu', action='store_true')
    args = parser.parse_args()

    if not args.force_gpu:
        autoGPU()

    main(args)

