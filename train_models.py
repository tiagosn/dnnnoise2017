import os
import sys
import numpy as np
import cPickle as pk

from bz2 import BZ2File

from networks import *

from keras.utils import np_utils

from utils import get_ds_full_name
from utils import dataset_info

gaussian_stds = [10, 20, 30, 40, 50]
sp_percentages = [10, 20, 30, 40, 50]
datasets = ['MNIST', 'CIFAR-10', 'SVHN']

def train_model(ds_name, noise_type=None, noise_level=None, denoised=False):
    ds_full_name = get_ds_full_name(ds_name, noise_type, noise_level, denoised)
    ds_file = ds_full_name + '.bz2'

    model_out_file = 'net_%s.h5' % (ds_full_name)
    if os.path.exists(model_out_file):
        print '%s already exists!!!' % (model_out_file)
        return

    if not os.path.exists(ds_file):
        print 'Dataset %s does not exist!!!' % (ds_file)
        return

    print 'Loading %s ...' % (ds_file),
    sys.stdout.flush()
    ds = pk.load(BZ2File(ds_file, 'rb'))
    print ' Done!!!'

    X_train = np.double(ds['train']['X'])/255.0
    y_train = np_utils.to_categorical(ds['train']['y'], dataset_info[ds_name]['n_classes'])
    X_test = np.double(ds['test']['X'])/255.0
    y_test = np_utils.to_categorical(ds['test']['y'], dataset_info[ds_name]['n_classes'])

    print 'Training model:'
    model = dataset_info[ds_name]['net'](dataset_info[ds_name]['im_shape'], dataset_info[ds_name]['n_classes'])
    model.fit(X_train, y_train, batch_size=dataset_info[ds_name]['batch_size'], nb_epoch=dataset_info[ds_name]['n_epochs'],
          verbose=1, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)

    model.save_weights(model_out_file)

    print '\t[net_%s] Acc on test: %2.2lf%%' % (ds_full_name, 100.0*score[1])

for d in datasets:
    train_model(d) # original (without noise)

    for p in sp_percentages:
        train_model(d, 'sp', p)
        train_model(d, 'sp', p, True)

    for s in gaussian_stds:
        train_model(d, 'gaussian', s)
        train_model(d, 'gaussian', s, True)
