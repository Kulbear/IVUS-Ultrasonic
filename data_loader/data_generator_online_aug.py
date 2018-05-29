import os.path as path

import numpy as np
from .image_augmentor import ImageAugmentor

data_dir = 'NPY_FILES'
use_deconv = 'D_'

if use_deconv:
    data_dir = use_deconv + data_dir

cdir = '..'
nb_source_img = 6


def shuffle_lists(listx, listy):
    xy = list(zip(listx, listy))
    np.random.shuffle(xy)
    final_x, final_y = zip(*xy)
    return list(final_x), list(final_y)


class IVUSDataGenerator:
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()
        self.input_batches = []
        self.mask_batches = []
        self.counter = 0

        # load data here
        print('Load data for the {} model...'.format(config.target))
        self.input = np.expand_dims(np.load(
            path.join(path.abspath(cdir), data_dir, 'raw_train_data_{}.npy'.format(img_size))), -1)
        self.y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'raw_train_{}_labels_512.npy'.format(target)))
        print(self.input.shape)
        print(self.y.shape)

        self.test_input = np.expand_dims(
            np.load(
                path.join(path.abspath(cdir), data_dir, 'test_data_{}.npy'.format(img_size))),
            -1)
        self.test_y = np.load(
            path.join(
                path.abspath(cdir), data_dir,
                'test_{}_labels_512.npy'.format(target))).astype(np.uint8)
        print(self.test_input.shape)
        print(self.test_y.shape)

    def next_batch(self, batch_size=10):
        # reset batches when next epoch
        self.counter += 1
        if self.counter == self.config.num_iter_per_epoch // 3:
            print('Drop exisiting batches...\n')
            self.input_batches = []
            self.mask_batches = []
            self.counter = 0

        if len(self.input_batches):
            # provide one group of result
            input_batch = self.input_batches.pop()
            input_y = self.mask_batches.pop()
            return np.array(input_batch), np.array(input_y)

        else:
            print('Generating new batches...')
            self.counter == 0
            self.input_batches = []
            self.mask_batches = []
            idx = np.random.choice(self.input.shape[0], nb_source_img, replace=False)
            aug = ImageAugmentor(self.input[idx], mode='SIMPLE', labels=self.y[idx], end_to_end=True)
            aug_x, aug_y = aug.generate()
            aug_x = list(aug_x[:aug_x.shape[0] // 5 * 3])  # select the first 3/5
            aug_y = list(aug_y[:aug_y.shape[0] // 5 * 3])  # the second 1/2 are all black images

            aug_x, aug_y = shuffle_lists(aug_x, aug_y)

            for i in range(0, len(aug_x), batch_size):
                self.input_batches.append(aug_x[i:i + batch_size])
                self.mask_batches.append(aug_y[i:i + batch_size])

            self.input_batches.pop()
            self.mask_batches.pop()

            # provide one group of result
            input_batch = self.input_batches.pop()
            input_y = self.mask_batches.pop()
            return np.array(input_batch), np.array(input_y)

    def get_test_subsets(self, subset_size):
        test_set_size = self.test_input.shape[0]
        test_subsets = [(self.test_input[i:i + subset_size],
                         self.test_y[i:i + subset_size])
                        for i in range(0, test_set_size, subset_size)]
        return test_subsets
