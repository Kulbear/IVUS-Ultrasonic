import numpy as np
import os.path as path

data_dir = 'D_NPY_FILES'


class IVUSDataGenerator():
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()

        # load data here
        print('Load data for the {} model...'.format(config.target))
        self.input = np.expand_dims(
            np.load(
                path.join(
                    path.abspath('..'), data_dir,
                    'aug_train_data_{}.npy'.format(img_size))), -1)
        print(self.input.shape)
        self.y = np.load(
            path.join(
                path.abspath('..'), data_dir,
                'aug_train_{}_labels_{}.npy'.format(target, img_size)))
        print(self.y.shape)

        self.test_input = np.expand_dims(
            np.load(
                path.join(
                    path.abspath('..'), data_dir,
                    'test_data_{}.npy'.format(img_size))), -1)
        self.test_y = np.load(
            path.join(
                path.abspath('..'), data_dir, 'test_{}_labels_{}.npy'.format(
                    target, img_size))).astype(int)

        print(self.test_input.shape)
        print(self.test_y.shape)

    def next_batch(self, batch_size=5):
        idx = np.random.choice(self.input.shape[0], batch_size, replace=False)
        yield self.input[idx], self.y[idx]

    def get_test_subsets(self, subset_size):
        test_set_size = self.test_input.shape[0]
        test_subsets = [(self.test_input[i:i + subset_size],
                         self.test_y[i:i + subset_size])
                        for i in range(0, test_set_size, subset_size)]
        return test_subsets
