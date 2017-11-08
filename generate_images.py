import os
import numpy as np
import skimage.io as io
import cPickle as pk

from bz2 import BZ2File

ds_file = 'MNIST_sp-20_denoised.bz2'
out_folder = 'ims_MNIST_sp-20_denoised'

ds = pk.load(BZ2File(ds_file, 'rb'))

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

n_channels = ds['train']['X'].shape[1]
for i in xrange(0, ds['train']['X'].shape[0]):
    if n_channels == 3:
        im = np.transpose(ds['train']['X'][i], axes=(1,2,0))
    else:
        im = ds['train']['X'][i][0]

    out_file_path = '%s/train%d.png' % (out_folder, i)
    io.imsave(out_file_path, im)
