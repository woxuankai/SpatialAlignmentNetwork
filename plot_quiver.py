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

def bias_correction(img):
    return img
    img = sitk.GetImageFromArray(img.astype(np.float64))
    #img = sitk.N4BiasFieldCorrection(img, convergenceThreshold = 1e-4)
    img = sitk.N4BiasFieldCorrection(img)
    return sitk.GetArrayFromImage(img)

def checker_borad(A, B, block):
    assert A.shape == B.shape
    assert len(A.shape) == 2
    i, j = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    i, j = (i+block//2)//block%2, (j+block//2)//block%2,
    mask = np.logical_xor(i, j)
    C = mask*A + np.logical_not(mask)*B
    return C

print('usage: '+sys.argv[0]+' img sampled rec ref warped offset ind [save]')
img = nib.load(sys.argv[1]).get_fdata().T
sampled = nib.load(sys.argv[2]).get_fdata().T
rec = nib.load(sys.argv[3]).get_fdata().T
ref = nib.load(sys.argv[4]).get_fdata().T
warped = nib.load(sys.argv[5]).get_fdata().T
offset = nib.load(sys.argv[6]).get_fdata().T
ind = int(sys.argv[7])
if len(sys.argv)==9:
    save_path = sys.argv[8]
else:
    save_path = None

# PSNR and SSIM
from metrics import psnr, ssim
print('PSRN: '+str(psnr(img[:,None,...], rec[:,None,...])))
print('SSIM: '+str(ssim(img[:,None,...], rec[:,None,...])))

_dX, _dY, _ = map(lambda x: x[0,ind], offset)
ref, sampled, rec, img, warped = ref[ind], sampled[ind], rec[ind], img[ind], warped[ind]

y_len, x_len = img.shape
ratio = 10
scale = 1./4
block = 40
vmax = 0.9

dX = scipy.ndimage.gaussian_filter(_dX, sigma=3)[ratio//2:y_len:ratio, ratio//2:x_len:ratio]
dY = scipy.ndimage.gaussian_filter(_dY, sigma=3)[ratio//2:y_len:ratio, ratio//2:x_len:ratio]
Y, X = np.mgrid[ratio//2:y_len:ratio, ratio//2:x_len:ratio]

x_major_ticks = np.arange((x_len%block)//2, x_len, block)
y_major_ticks = np.arange((y_len%block)//2, y_len, block)
#ref, img, warped = map(bias_correction, (ref[ind], img[ind], warped[ind]))

img_warped = checker_borad(img, warped, block)
img_ref = checker_borad(img, ref, block)

# plot
fig = plt.figure(figsize=[10, 10])
axs = fig.subplots(3, 3)
for ax in axs.flatten():
    ax.set_xticks(x_major_ticks)
    ax.set_yticks(y_major_ticks)
    ax.set_box_aspect(1)

def tick_and_grid(ax, grid=None, ticks=None):
    if grid is not None:
        ax.grid(grid)
    if ticks is not None:
        #ax.tick_params(which='both', length=0)
        ax.tick_params(which='both', \
                bottom=False, top=False, \
                left=False, right=False)
        ax.tick_params(which='both', \
                labelbottom=False, labeltop=False, \
                labelleft=False, labelright=False)
        #ax.tick_params(axis='x', colors=(0,0,0,0))
        #ax.tick_params(axis='y', colors=(0,0,0,0))
        ax.invert_yaxis()
        ax.invert_xaxis()

ax = axs[0, 0]
ax.imshow(ref, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Reference')
tick_and_grid(ax, grid=True, ticks=False)

ax = axs[0, 1]
ax.imshow(img, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Full Target')
tick_and_grid(ax, grid=True, ticks=False)

ax = axs[0, 2]
ax.imshow(warped, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Aligned Reference')
tick_and_grid(ax, grid=True, ticks=False)

ax = axs[1, 0]
ax.imshow(img_ref, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Target - Reference')
tick_and_grid(ax, grid=False, ticks=False)

ax = axs[1, 1]
ax.imshow(rec, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Reconstruction')
tick_and_grid(ax, grid=True, ticks=False)

ax = axs[1, 2]
ax.imshow(img_warped, cmap='gray', vmax=vmax, interpolation='none')
ax.set_title('Target - Aligned Reference')
tick_and_grid(ax, grid=False, ticks=False)

ax = axs[2,0]
#ax.imshow(sampled, cmap='gray', vmax=vmax)
#ax.set_title('Partial Target')
ax.imshow(warped, cmap='gray', vmax=vmax, interpolation='none')
ax.quiver(X, Y, dX, dY, \
        color='yellow', units='xy', angles='xy', scale_units='xy', scale=scale)
ax.set_title('Displacement (x'+str(int(1/scale))+')')
tick_and_grid(ax, grid=True, ticks=False)

ax = axs[2,1]
err = ax.imshow(np.abs(rec-img), cmap='plasma', vmin=0, vmax=0.3, interpolation='none')
ax.set_title('Reconstruction Error')
tick_and_grid(ax, grid=False, ticks=False)

ax = axs[2,2]
#err = ax.imshow(rec-img, cmap='bwr', vmin=-0.1, vmax=0.1)
#ax.set_title('Reconstruction Error')
#tick_and_grid(ax, grid=True, ticks=False)
fig.colorbar(err, ax=ax)

if save_path is not None:
    plt.savefig(save_path)


fig = plt.figure(figsize=[9, 3])
axs = fig.subplots(1, 3)

ax = axs[0]
ax.imshow(np.zeros_like(ref), cmap='gray', vmin=0, vmax=1, interpolation='none')
scatter = ax.quiver(X, Y, dX, dY, color='yellow', units='xy', angles='xy', scale_units='xy', scale=scale)
tick_and_grid(ax, grid=False, ticks=False)

ax = axs[1]
sX, sY = np.meshgrid(np.linspace(0, 1, x_len), np.linspace(0, 1, y_len))
RGB = cv.cvtColor(np.stack((np.zeros_like(sX)+0.5, 1-sY, sX), axis=-1).astype(np.float32), cv.COLOR_YUV2RGB)
rgb_img = ax.imshow(np.clip(RGB, 0, 1), interpolation='none')
tick_and_grid(ax, grid=False, ticks=False)

ax = axs[2]
RGB = cv.cvtColor(((np.clip(np.stack((np.zeros_like(_dX), -_dY, _dX), axis=-1), -4, 4)+4)/8).astype(np.float32), cv.COLOR_YUV2RGB)
rgb_img = ax.imshow(np.clip(RGB, 0, 1), interpolation='none')
tick_and_grid(ax, grid=False, ticks=False)

# plt.figure()
# rgb_img = plt.imshow((_dX**2+_dY**2)**0.5)
# ax = rgb_img.axes
# plt.colorbar()
# ax.grid(True)
# ax.set_box_aspect(1)

plt.tight_layout()
plt.show()


