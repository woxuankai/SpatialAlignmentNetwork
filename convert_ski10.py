#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import nibabel as nib
import h5py

def convert(path_in, label_in, path_out):
    img = nib.load(path_in)
    data = img.get_fdata().T
    minVal, maxVal = data.min(), data.max()
    data = (data - minVal)/(maxVal-minVal)
    #data = data/maxVal
    h5 = h5py.File(path_out, 'w')
    h5.attrs['max'] = 1
    h5['image'] = data.astype(np.complex64)
    if label_in is not None:
        label = nib.load(label_in)
        data = label.get_data().T
        h5['label'] = data
    h5.close()
    
if __name__ == '__main__':
    path_in, label_in, path_out = sys.argv[1], sys.argv[2], sys.argv[3]
    if label_in == "":
        label_in = None
    convert(path_in, label_in, path_out)

