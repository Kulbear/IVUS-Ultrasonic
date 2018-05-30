import os.path as path

import numpy as np
from .image_augmentor import ImageAugmentor

cdir = '..'


class IVUSDataGenerator:
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()
        self.input_batches = []
        self.mask_batches = []
        self.counter = 0
        data_dir = 'processed_augmentations'
        data_dir = path.join(data_dir, config.dir)
        print(data_dir)

        # load data here
        print('Load data for the {} model...'.format(config.target))
        self.input = np.expand_dims(np.load(
            path.join(path.abspath(cdir), data_dir, 'train_img_{}.npy'.format(img_size))), -1)
        self.y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'train_{}_512.npy'.format(target)))
        print(self.input.shape)
        print(self.y.shape)

        self.test_input = np.expand_dims(
            np.load(
                path.join(path.abspath(cdir), data_dir, 'test_img_{}.npy'.format(img_size))),
            -1)
        self.test_y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'test_{}_512.npy'.format(target))).astype(np.uint8)
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
