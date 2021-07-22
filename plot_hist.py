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

def hist(datasets, labels, bins, max_count):
    binned_data_sets = [np.histogram(d, bins=bins)[0] for d in datasets]
    #binned_maximums = np.max(binned_data_sets, axis=1)
    binned_maximums = [max_count]*len(labels)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
    centers = 0.5 * (bins + np.roll(bins, 1))[:-1]
    heights = np.diff(bins)
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts)
    ax.set_xticks(x_locations)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Data values")
    ax.set_xlabel("Data sets")
    ax.plot((min(x_locations)-max_count/2., max(x_locations)+max_count/2.),(0,0),'k--')

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
    step = 0.2
    labels = ('reg', 'rec', 'rec-reg')
    datasets = (psnrReg - psnrNone, psnrRec - psnrNone, psnrMixed - psnrNone)
    minVal, maxVal = np.min(datasets), np.max(datasets)
    minmax = (-0.6,2.0)
    print(minVal, maxVal)
    #assert minmax[0] <= minVal
    #assert minmax[1] >= maxVal
    #minBin, maxBin = minVal - minVal%step, maxVal + maxVal%step
    #bins = np.arange(minBin, maxBin+step, step)
    bins =  np.arange(minmax[0], minmax[1], step)
    #print(bins)
    hist(datasets, labels, bins, max_count=60)
    #ax.plot(minmax, minmax)
    ax.set_ylim(minmax)
    #ax.set_xlabel('rec-reg')
    #ax.set_ylim(minmax)
    #ax.set_ylabel('none/reg/rec')
    #ax.legend()
    ax.set_title('PSNR')
    #ax.set_box_aspect(1)

    ax = axs[1]
    step = 0.001
    datasets = (ssimReg - ssimNone, ssimRec - ssimNone, ssimMixed - ssimNone)
    minVal, maxVal = np.min(datasets), np.max(datasets)
    #minBin, maxBin = minVal - minVal%step, maxVal + maxVal%step
    #bins = np.arange(minBin, maxBin+step, step)
    print(minVal, maxVal)
    minmax = (-0.004, 0.014)
    bins = np.arange(minmax[0], minmax[1], step)
    print(bins)
    hist(datasets, labels, bins, max_count=60)
    #ax.plot(minmax, minmax)
    ax.set_ylim(minmax)
    #ax.set_xlabel('rec-reg')
    #ax.set_ylim(minmax)
    #ax.set_ylabel('none/reg/rec')
    ax.set_title('SSIM')

    plt.show()


    exit()

    area = 5
    ax = axs[0]
    ax.scatter(psnrMixed, psnrNone, s=area, label='rec-reg none')
    ax.scatter(psnrMixed, psnrReg, s=area, label='rec-reg reg')
    ax.scatter(psnrMixed, psnrRec, s=area, label='rec-reg rec')
    minmax = (29,41)
    ax.plot(minmax, minmax)
    ax.set_xlim(minmax)
    ax.set_xlabel('rec-reg')
    ax.set_ylim(minmax)
    ax.set_ylabel('none/reg/rec')
    ax.legend()
    ax.set_title('PSNR')
    ax.set_box_aspect(1)

    ax = axs[1]
    ax.scatter(ssimMixed, ssimNone, s=area, label='rec-reg none')
    ax.scatter(ssimMixed, ssimReg, s=area, label='rec-reg reg')
    ax.scatter(ssimMixed, ssimRec, s=area, label='rec-reg rec')
    minmax = (0.90,0.985)
    ax.plot(minmax, minmax)
    ax.set_xlim(minmax)
    ax.set_xlabel('rec-reg')
    ax.set_ylim(minmax)
    ax.set_ylabel('none/reg/rec')
    ax.legend()
    ax.set_title('SSIM')
    ax.set_box_aspect(1)

    plt.show()
    exit()

    
