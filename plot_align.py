import sys
import numpy as np
import scipy
import scipy.ndimage
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk

def checker_borad(A, B, block):
    assert A.shape == B.shape
    assert len(A.shape) == 2
    i, j = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    i, j = (i+block//2)//block%2, (j+block//2)//block%2,
    mask = np.logical_xor(i, j)
    C = mask*A + np.logical_not(mask)*B
    return C

print('usage: '+sys.argv[0]+', ind, save, gt, ref, warped1, offset1, warped2, offset2, warped3, offset3')
ind = int(sys.argv[1])
save_path = sys.argv[2] if sys.argv[2] != 'None' else None
img = nib.load(sys.argv[3]).get_fdata().T
ref = nib.load(sys.argv[4]).get_fdata().T
warpeds = [nib.load(i).get_fdata().T for i in sys.argv[5::2]]
offsets = [nib.load(i).get_fdata().T for i in sys.argv[6::2]]

img, ref = img[ind], ref[ind]
warpeds = [warped[ind] for warped in warpeds]
offsets = [offset[:, 0, ind] for offset in offsets]

ratio = 10
scale = 1./4
block = 40
vmax = 0.7

y_len, x_len = img.shape
Y, X = np.mgrid[ratio//2:y_len:ratio, ratio//2:x_len:ratio]
x_major_ticks = np.arange(block//2, x_len, block)
y_major_ticks = np.arange(block//2, y_len, block)

# plot
fig = plt.figure(figsize=[10, 8])
axs = fig.subplots(5, len(warpeds))

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

#ref, img, warped = map(bias_correction, (ref[ind], img[ind], warped[ind]))
for i, (warped, offset) in enumerate(zip(warpeds, offsets)):
    _dX, _dY, _ = offset
    dX = scipy.ndimage.gaussian_filter(_dX, sigma=3)[\
            ratio//2:y_len:ratio, ratio//2:x_len:ratio]
    dY = scipy.ndimage.gaussian_filter(_dY, sigma=3)[\
            ratio//2:y_len:ratio, ratio//2:x_len:ratio]

    img_warped = checker_borad(img, warped, block)
    img_ref = checker_borad(img, ref, block)

    ax = axs[0, i]
    ax.imshow(ref, cmap='gray', vmax=vmax, interpolation='none')
    ax.set_title('Reference')
    tick_and_grid(ax, grid=True, ticks=False)

    ax = axs[1, i]
    ax.imshow(img, cmap='gray', vmax=vmax, interpolation='none')
    ax.set_title('Full Target')
    tick_and_grid(ax, grid=True, ticks=False)

    ax = axs[2, i]
    ax.imshow(warped, cmap='gray', vmax=vmax, interpolation='none')
    ax.set_title('Aligned Reference')
    tick_and_grid(ax, grid=True, ticks=False)

    ax = axs[3, i]
    ax.imshow(img_warped, cmap='gray', vmax=vmax, interpolation='none')
    ax.set_title('Target - Aligned Reference')
    tick_and_grid(ax, grid=False, ticks=False)

    ax = axs[4,i]
    ax.imshow(warped, cmap='gray', vmax=vmax, interpolation='none')
    ax.quiver(X, Y, dX, dY, \
            color='yellow', units='xy', angles='xy', \
            scale_units='xy', scale=scale)
    ax.set_title('Displacement (x'+str(int(1/scale))+')')
    tick_and_grid(ax, grid=True, ticks=False)

for ax in axs.flatten():
    ax.set_xticks(x_major_ticks)
    ax.set_yticks(y_major_ticks)
    ax.set_box_aspect(1)
    ax.set_title('')

if save_path is not None:
    plt.savefig(save_path)

plt.show()



