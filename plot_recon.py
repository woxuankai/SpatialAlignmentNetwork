import sys
import numpy as np
import scipy
import scipy.ndimage
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import cv2 as cv

print('usage: '+sys.argv[0]+' ind, save, gt rec1, rec2, rec3')
ind = int(sys.argv[1])
save_path = sys.argv[2] if sys.argv[2] != 'None' else None
img = nib.load(sys.argv[3]).get_fdata().T
recons = [nib.load(recon).get_fdata().T for recon in sys.argv[4:]]

# PSNR and SSIM
from metrics import psnr, ssim
for rec in recons:
    print('PSRN: '+str(psnr(img[:,None,...], rec[:,None,...])) \
            + ' SSIM: '+str(ssim(img[:,None,...], rec[:,None,...])))

img, *recons = [x[ind] for x in (img, *recons)]
y_len, x_len = img.shape
vmax = 0.7
emax = 0.2

# plot
fig = plt.figure(figsize=[10, 4])
axs = fig.subplots(2, len(recons)+1)

for i, rec in enumerate(recons):
    ax = axs[0, i]
    ax.imshow(rec, cmap='gray', vmin=0, vmax=vmax, interpolation='none')
    # ax.set_title('Reference')
    ax.set_box_aspect(1)

    ax = axs[1, i]
    err = ax.imshow(np.abs(rec-img), \
            cmap='viridis', vmin=0, vmax=emax, interpolation='none')
    # ax.set_title('Reconstruction Error')
    ax.set_box_aspect(1)

ax = axs[0, len(recons)]
ax.imshow(img, cmap='gray', vmin=0, vmax=vmax, interpolation='none')
ax.set_title('GT')

ax = axs[1,len(recons)]
ax.imshow(np.ones_like(img), \
        cmap='gray', vmin=0, vmax=vmax, interpolation='none')
fig.colorbar(err, ax=ax)

for ax in axs.flatten():
    #ax.patch.set_visible(False)
    ax.invert_yaxis()
    ax.invert_xaxis()
    #ax.tick_params(which='both', \
    #        bottom=False, top=False, \
    #        left=False, right=False)
    #ax.tick_params(which='both', \
    #        labelbottom=False, labeltop=False, \
    #        labelleft=False, labelright=False)
    ax.axis('off')

if save_path is not None:
    plt.savefig(save_path)

plt.show()


