import os
import sys
import numpy as np
import cPickle as pk
import skimage as si

from bz2 import BZ2File

def std2var(std):
    norm_std = std/255.0
    var = norm_std**2

    return var

def var2std(var):
    std = np.sqrt(var) * 255.0
    std = int(np.round(std))

    return std

def params2str(params):
    noise_type = 'sp' if params['mode'] == 's&p' else params['mode']
    noise_degree = int(params['amount']*100) if noise_type == 'sp' else var2std(params['var'])

    out_str = '%s-%d' % (noise_type, noise_degree)

    return out_str

def add_noise(data, noise_params):
    for i in xrange(0, data.shape[0]): # for every image
        for j in xrange(0, data.shape[1]): # for every channel
            channel = np.double(data[i,j])/255.0
            channel = si.util.random_noise(channel, **noise_params)
            data[i,j] = np.uint8(channel*255.0)

    return data

def create_noise_dataset(ds_name, noise_params):
    str_params = params2str(noise_params)

    out_file = '%s_%s.bz2' % (ds_name, str_params)
    if os.path.exists(out_file):
        print '%s already exists!!!' % (out_file)
        return

    print 'Processing %s dataset (%s):' % (ds_name, str_params)

    ds_file = ds_name + '_original.bz2'
    print '\t Loading %s ...' % (ds_file),
    sys.stdout.flush()
    ds = pk.load(BZ2File(ds_file, 'rb'))
    print ' Done!!!'

    print '\t Adding noise to training set ...',
    sys.stdout.flush()
    ds['train']['X'] = add_noise(ds['train']['X'], noise_params)
    print ' Done!!!'

    print '\t Adding noise to test set ...',
    sys.stdout.flush()
    ds['test']['X'] = add_noise(ds['test']['X'], noise_params)
    print ' Done!!!'

    print '\t Saving %s ...' % (out_file),
    sys.stdout.flush()
    pk.dump(ds, BZ2File(out_file, 'wb'))
    print ' Done!!!'
    print


noiseTypes = [
        {'mode': 'gaussian', 'var': std2var(10)},
        {'mode': 'gaussian', 'var': std2var(20)},
        {'mode': 'gaussian', 'var': std2var(30)},
        {'mode': 'gaussian', 'var': std2var(40)},
        {'mode': 'gaussian', 'var': std2var(50)},
        {'mode': 's&p',      'amount': 0.1     },
        {'mode': 's&p',      'amount': 0.2     },
        {'mode': 's&p',      'amount': 0.3     },
        {'mode': 's&p',      'amount': 0.4     },
        {'mode': 's&p',      'amount': 0.5     }
]

datasets = ['MNIST', 'CIFAR-10', 'SVHN']

for d in datasets:
    for n in noiseTypes:
        create_noise_dataset(d, n)
