import sys
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import cPickle as pk

from bz2 import BZ2File
from keras.utils import np_utils

gaussian_sds = [10, 20, 30, 40, 50]
sp_percentages = [10, 20, 30, 40, 50]
datasets = ['MNIST', 'CIFAR-10', 'SVHN']

n_configs = 2*len(gaussian_sds) + 2*len(sp_percentages) + 1

def load_ds(ds_file):
    print 'Loading %s ...' % (ds_file),
    sys.stdout.flush()
    ds = pk.load(BZ2File(ds_file, 'rb'))
    print ' Done!!!'

    return ds

def load_model(model_file):
    print 'Loading model %s ...' % (model_file),
    sys.stdout.flush()
    model = ut.dataset_info[ds_name]['net'](ut.dataset_info[ds_name]['im_shape'], ut.dataset_info[ds_name]['n_classes'])
    model.load_weights(model_file)
    print ' Done!!!'

    return model

def classify(ds, ds_name):
    out = np.zeros((n_configs))
    X_test = np.double(ds['test']['X'])/255.0
    y_test = np_utils.to_categorical(ds['test']['y'], ut.dataset_info[ds_name]['n_classes'])

    all_ds = ut.get_all_ds_names(ds_name)
    for i in xrange(0, len(all_ds)):
        model = load_model('net_' + all_ds[i] + '.h5')
        score = model.evaluate(X_test, y_test, verbose=0)
        out[i] = score[1]

    return out

def print_heatmap(mat):
    pass

for ds_name in datasets:
    all_ds = ut.get_all_ds_names(ds_name)
    heatmap_mat = np.zeros((n_configs, n_configs))

    for i in xrange(0, len(all_ds)):
        ds = load_ds(all_ds[i] + '.bz2')
        heatmap_mat[:,i] = classify(ds, ds_name)

    out_file = ds_name + '-results.bz2'
    pk.dump(heatmap_mat, BZ2File(out_file, 'wb'))
    print heatmap_mat
