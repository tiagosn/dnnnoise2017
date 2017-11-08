from networks import *

dataset_info = {
    'MNIST': {'net': almost_LeNet, 'n_classes': 10, 'im_shape': (1,28,28), 'batch_size': 128, 'n_epochs': 50},
    'CIFAR-10': {'net': all_conv_net_ref_c, 'n_classes': 10, 'im_shape': (3,32,32), 'batch_size': 128, 'n_epochs': 100},
    'SVHN': {'net': all_conv_net_ref_c, 'n_classes': 10, 'im_shape': (3,32,32), 'batch_size': 128, 'n_epochs': 100}
}

gaussian_sds = [10, 20, 30, 40, 50]
sp_percentages = [10, 20, 30, 40, 50]
datasets = ['MNIST', 'CIFAR-10', 'SVHN']

def get_ds_full_name(ds_name, noise_type=None, noise_level=None, denoised=False):
    str_type = noise_type if noise_type is not None else 'original'
    str_level = '-%d' % (noise_level) if noise_level is not None else ''
    str_denoised = '_denoised' if denoised else ''

    str_out = '%s_%s%s%s' % (ds_name, str_type, str_level, str_denoised)

    return str_out

def get_all_ds_names(ds_name):
    names = []

    names.append(get_ds_full_name(ds_name))

    for s in gaussian_sds:
        names.append(get_ds_full_name(ds_name, 'gaussian', s))

    for p in sp_percentages:
        names.append(get_ds_full_name(ds_name, 'sp', p))

    for s in gaussian_sds:
        names.append(get_ds_full_name(ds_name, 'gaussian', s, True))

    for p in sp_percentages:
        names.append(get_ds_full_name(ds_name, 'sp', p, True))

    return names
