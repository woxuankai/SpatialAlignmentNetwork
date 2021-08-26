import sys, os
import h5py
import numpy as np
import nibabel as nib

def convert(nii_path, h5_path, protocal):
    # convert nii file with path nii_path to h5 file stored at h5_path
    # protocal name as string
    h5 = h5py.File(h5_path, 'w')
    nii = nib.load(nii_path)
    array = nib.as_closest_canonical(nii).get_fdata() #convert to RAS
    #array = array.astype(np.float32)
    array = array.T.astype(np.float32)
    #h5['image'] = array#*(1+0j)
    h5.create_dataset('image', data=array)
    h5.attrs['max'] = array.max()
    h5.attrs['acquisition'] = protocal
    h5.close()

if __name__ == '__main__':
    convert(nii_path=sys.argv[1], h5_path=sys.argv[2], protocal=sys.argv[3])
