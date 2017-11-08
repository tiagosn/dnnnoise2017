import os
import sys
import wget
import scipy.io as sio
import cPickle as pk
import numpy as np

from bz2 import BZ2File

from keras.datasets import cifar10, mnist

# All images are a 4-D np.array with the following dimensions (n_channels, n_rows, n_cols)

# Create SVHN dataset
# download train and test set (if necessary)
def create_SVHN():
    if os.path.exists('SVHN_original.bz2'):
        print 'SVHN dataset already exists!!!'
        return

    if not os.path.exists('SVHN_train.mat'):
        print 'Downloading SVHN training set:'
        wget.download('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', out='SVHN_train.mat')
        print

    if not os.path.exists('SVHN_test.mat'):
        print 'Downloading SVHN test set:'
        wget.download('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', out='SVHN_test.mat')
        print

    print 'Creating SVHN dataset ...',
    sys.stdout.flush()
    SVHN_train_mat = sio.loadmat('SVHN_train.mat')
    SVHN_test_mat = sio.loadmat('SVHN_test.mat')

    # images of zeros are class 10 (MATLAB)
    SVHN_train_mat['y'][SVHN_train_mat['y'] == 10] = 0
    SVHN_test_mat['y'][SVHN_test_mat['y'] == 10] = 0

    SVHN_original = {}
    SVHN_original['train'] = {}
    SVHN_original['test'] = {}
    SVHN_original['train']['X'] = np.transpose(SVHN_train_mat['X'], axes=(3,2,0,1))
    SVHN_original['train']['y'] = SVHN_train_mat['y'].copy()
    SVHN_original['test']['X'] = np.transpose(SVHN_test_mat['X'], axes=(3,2,0,1))
    SVHN_original['test']['y'] = SVHN_test_mat['y'].copy()

    pk.dump(SVHN_original, BZ2File('SVHN_original.bz2', 'wb'))
    print ' Done!!!'

# Create CIFAR-10 dataset
def create_CIFAR10():
    if os.path.exists('CIFAR-10_original.bz2'):
        print 'CIFAR-10 dataset already exists!!!'
        return

    print 'Creating CIFAR-10 dataset ...',
    sys.stdout.flush()
    cifar10_original = {}
    cifar10_original['train'] = {}
    cifar10_original['test'] = {}

    (cifar10_original['train']['X'], cifar10_original['train']['y']), \
    (cifar10_original['test']['X'], cifar10_original['test']['y'])\
        = cifar10.load_data()

    pk.dump(cifar10_original, BZ2File('CIFAR-10_original.bz2', 'wb'))
    print ' Done!!!'

# Create MNIST dataset

def create_MNIST():
    if os.path.exists('MNIST_original.bz2'):
        print 'MNIST dataset already exists!!!'
        return

    print 'Creating MNIST dataset ...',
    sys.stdout.flush()
    mnist_original = {}
    mnist_original['train'] = {}
    mnist_original['test'] = {}

    (mnist_original['train']['X'], mnist_original['train']['y']), \
    (mnist_original['test']['X'], mnist_original['test']['y'])\
        = mnist.load_data()

    # add a dim for the single (gray) channel
    # shape: (n_samples, 28, 28) -> (n_samples, 1, 28, 28)
    train_shape = mnist_original['train']['X'].shape
    mnist_original['train']['X'] = np.reshape(mnist_original['train']['X'],\
                                                (train_shape[0], 1, train_shape[1], train_shape[2]))
    test_shape = mnist_original['test']['X'].shape
    mnist_original['test']['X'] = np.reshape(mnist_original['test']['X'],\
                                                (test_shape[0], 1, test_shape[1], test_shape[2]))

    pk.dump(mnist_original, BZ2File('MNIST_original.bz2', 'wb'))
    print ' Done!!!'

create_SVHN()
create_CIFAR10()
create_MNIST()
