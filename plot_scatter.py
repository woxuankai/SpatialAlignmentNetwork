import sys
import json
import numpy as np
import scipy
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt


def parseJSON(f):
    with open(f) as f:
        f = json.load(f)
        psnr = np.array([x['metric_PSNR'] for x in f])
        ssim = np.array([x['metric_SSIM'] for x in f])
    return psnr, ssim

if __name__ == '__main__':
    print('usage: '+sys.argv[0]+' none reg rec rec-reg [save]')
    assert len(sys.argv) <= 6
    psnrNone, ssimNone = parseJSON(sys.argv[1])
    psnrReg, ssimReg = parseJSON(sys.argv[2])
    psnrRec, ssimRec = parseJSON(sys.argv[3])
    psnrMixed, ssimMixed = parseJSON(sys.argv[4])
    save = sys.argv[5] if len(sys.argv)==6 else None

    fig = plt.figure(figsize=[8, 4])
    axs = fig.subplots(1, 2)

    ax = axs[0]
    labels = ('none', 'reg', 'rec', 'rec-reg')
    x_locs = range(len(labels))
    datasets = np.array([psnrNone, psnrReg, psnrRec, psnrMixed]).T
    for dataset in datasets:
        ax.plot(x_locs, dataset, 'o--')
    ax.set_xticks(x_locs)
    ax.set_xticklabels(labels)
    ax.set_title('PSNR')

    ax = axs[1]
    datasets = np.array([ssimNone, ssimReg, ssimRec, ssimMixed]).T
    for dataset in datasets:
        ax.plot(x_locs, dataset, 'o--')
    ax.set_xticks(x_locs)
    ax.set_xticklabels(labels)
    ax.set_title('SSIM')
    plt.show()
