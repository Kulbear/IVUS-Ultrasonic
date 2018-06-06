import os.path as path

import numpy as np
from .image_augmentor import ImageAugmentor

cdir = '..'


class IVUSDataGenerator:
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()

        self.data_format = self.config.data_format
        expanded_dim = 1 if self.data_format == 'NCHW' else -1
        data_dir = 'processed_augmentations'
        data_dir = path.join(data_dir, config.dir)
        print('[INFO] :: Reading data from ->', data_dir)

        # load train and test set and we split 1/10 of the train set to be the validation set
        print('[INFO] :: Loading data for the {} model...'.format(
            config.target))
        self.input = np.expand_dims(
            np.load(
                path.join(
                    path.abspath(cdir), data_dir, 'train_img_{}.npy'.format(
                        img_size))), expanded_dim).astype(np.float32) / 255
        self.y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'train_{}_512.npy'.format(target)))
        val_size = int(self.input.shape[0] // 10 * 9)
        self.val_input = self.input[val_size:]
        self.val_y = self.y[val_size:]
        self.input = self.input[:val_size]
        self.y = self.y[:val_size]
        self.test_input = np.expand_dims(
            np.load(
                path.join(
                    path.abspath(cdir), data_dir, 'test_img_{}.npy'.format(
                        img_size))), expanded_dim).astype(np.float32) / 255
        self.test_y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'test_{}_512.npy'.format(target))).astype(np.uint8)

        print('[INFO] :: Data loaded...\nDatasets are splited to\n')
        print('Training...')
        print(self.input.shape)
        print(self.y.shape)
        print('Validation...')
        print(self.val_input.shape)
        print(self.val_y.shape)
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
