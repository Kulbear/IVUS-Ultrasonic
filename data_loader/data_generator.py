import os.path as path
import h5py
import os
import numpy as np
from .image_augmentor import ImageAugmentor

cdir = '..'

img_amount = 719

class IVUSDataGenerator:
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()
        source_dir = 'dataset/' + config.dir

        self.data_format = self.config.data_format
        expanded_dim = 1 if self.data_format == 'NCHW' else -1

        print('[INFO] :: Reading data from ->', source_dir)

        print('[INFO] :: Loading data for the {} model...'.format(
        config.target))
        # load train and test set and we split 1/10 of the train set to be the validation set
        switch_dim = lambda x: np.rollaxis(x, -1, -2)
    
        files = [_ for _ in os.listdir('../' + source_dir) if _[-3:] == 'mat']

        img_ = []
        mask_ = []
        for f_idx, fname in enumerate(files):
            print(fname)
            data_dir = fname.split('.')[0]
            data_name = fname.split('.')[0]
            with h5py.File(path.join('../', source_dir, fname), 'r') as file:
                keys = list((file[data_name]['train']))
                m = np.array(file[data_name]['train']['img']).shape[0]
                idx = np.random.choice(m, m // 4)
                for k in keys:
                    data = file[data_name]['train'][k]
                    data = switch_dim(np.array(data))[idx]
                    if k == target:
                        mask_.append(data)
                    elif k == 'img':
                        img_.append(data)

        self.input = np.expand_dims(np.vstack(img_).copy(), expanded_dim).astype(np.float32) / 255
        del img_
        self.y = np.vstack(mask_).copy()
        del mask_

        # val_size = -1
        # val_size = int(self.input.shape[0] // 10 * 9)

        # self.val_input = self.input[val_size:]
        # self.val_y = self.y[val_size:]
        # self.input = self.input[:val_size]
        # self.y = self.y[:val_size]
        self.test_input = np.expand_dims(
            np.load(
                path.join(
                    path.abspath(cdir), source_dir, 'img.npy')), expanded_dim).astype(np.float32) / 255
        self.test_y = np.load(
            path.join(
                path.abspath(cdir), source_dir,
                target+'.npy')).astype(np.uint8)

        print('[INFO] :: Data loaded...\nDatasets are splited to\n')
        print('Training...')
        print(self.input.shape)
        print(self.y.shape)
        # print('Validation...')
        # print(self.val_input.shape)
        # print(self.val_y.shape)
        print('Testing...')
        print(self.test_input.shape)
        print(self.test_y.shape)

    def next_batch(self, batch_size=10):
        idx = np.random.choice(self.input.shape[0], batch_size, replace=False)
        return self.input[idx], self.y[idx]

    def get_test_subsets(self, subset_size):
        test_set_size = self.test_input.shape[0]
        test_subsets = [(self.test_input[i:i + subset_size],
                         self.test_y[i:i + subset_size])
                        for i in range(0, test_set_size, subset_size)]
        return test_subsets
