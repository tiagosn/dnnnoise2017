import numpy as np
import cPickle as pk
import utils as ut

from bz2 import BZ2File
from skimage.measure import compare_psnr

def get_image(data):
    if data.shape[0] == 3:
        return np.transpose(data, axes=(1,2,0))

    return data[0]

def mean_psnr(ds_original, ds_noisy):
    X_train_original = ds_original['train']['X']
    X_test_original = ds_original['test']['X']

    X_train_noisy = ds_noisy['train']['X']
    X_test_noisy = ds_noisy['test']['X']

    n_train_ims = X_train_original.shape[0]
    n_test_ims = X_test_original.shape[0]

    v_psnr = np.zeros(n_train_ims + n_test_ims)

    for i in xrange(0, n_train_ims):
        im_original = get_image(X_train_original[i])
        im_noisy = get_image(X_train_noisy[i])

        v_psnr[i] = compare_psnr(im_original, im_noisy)

    for i in xrange(0, n_test_ims):
        im_original = get_image(X_test_original[i])
        im_noisy = get_image(X_test_noisy[i])

        v_psnr[i + n_train_ims] = compare_psnr(im_original, im_noisy)

    return np.mean(v_psnr)

for ds_name in ut.datasets:
    all_ds = ut.get_all_ds_names(ds_name)

    ds_original = pk.load(BZ2File(all_ds[0] + '.bz2', 'rb'))
    for i in xrange(1, len(all_ds)):
        ds_noisy = pk.load(BZ2File(all_ds[i] + '.bz2', 'rb'))
        psnr = mean_psnr(ds_original, ds_noisy)
        print '%s -> %lf' % (all_ds[i], psnr)
