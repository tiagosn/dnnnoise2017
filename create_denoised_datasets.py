import os
import sys
import numpy as np
import cPickle as pk

from bz2 import BZ2File

from skimage.restoration import denoise_nl_means

from skimage.morphology import square
from skimage.filters.rank import median

def denoise(data, noise_type, noise_level):
    if noise_type == 'gaussian':
        return denoise_gaussian(data, noise_level)

    return denoise_sp(data)

def denoise_gaussian(data, std):
    std_norm = std/255.0

    print

    for i in xrange(0, data.shape[0]): # for every image
        for j in xrange(0, data.shape[1]): # for every channel
            channel = np.double(data[i,j])/255.0
            channel = denoise_nl_means(channel, h=std_norm)
            data[i,j] = np.uint8(channel*255.0)

        if i % 1000 == 0:
            p = 100.0*i/data.shape[0]
            print '\t\tstatus: %2.2lf%% completed' % (p)

    return data

def denoise_sp(data):
    for i in xrange(0, data.shape[0]): # for every image
        for j in xrange(0, data.shape[1]): # for every channel
            data[i,j] = median(data[i,j], square(3))

    return data

def create_denoised_dataset(ds_name, noise_type, noise_level):
    ds_file = '%s_%s-%d.bz2' % (ds_name, noise_type, noise_level)

    out_file = '%s_%s-%d_denoised.bz2' % (ds_name, noise_type, noise_level)
    if os.path.exists(out_file):
        print '%s already exists!!!' % (out_file)
        return

    print 'Denoising %s:' % (ds_file)
    print '\t Loading %s ...' % (ds_file),
    sys.stdout.flush()
    ds = pk.load(BZ2File(ds_file, 'rb'))
    print ' Done!!!'

    print '\t Denoising training set ...',
    sys.stdout.flush()
    ds['train']['X'] = denoise(ds['train']['X'], noise_type, noise_level)
    print ' Done!!!'

    print '\t Denoising test set ...',
    sys.stdout.flush()
    ds['test']['X'] = denoise(ds['test']['X'], noise_type, noise_level)
    print ' Done!!!'

    print '\t Saving %s ...' % (out_file),
    sys.stdout.flush()
    pk.dump(ds, BZ2File(out_file, 'wb'))
    print ' Done!!!'
    print

gaussian_stds = [10, 20, 30, 40, 50]
sp_percentages = [10, 20, 30, 40, 50]
datasets = ['MNIST', 'CIFAR-10', 'SVHN']

for d in datasets:
    for p in sp_percentages:
        create_denoised_dataset(d, 'sp', p)

    for s in gaussian_stds:
        create_denoised_dataset(d, 'gaussian', s)
