import os.path as path
import h5py
import os
import gc
import numpy as np

cdir = '..'

class IVUSDataGenerator:
    def __init__(self, config):
        self.config = config
        img_size = config.state_size[0]
        target = config.target.lower()
        source_dir = 'dataset/' + config.dir

        self.data_format = self.config.data_format
        expanded_dim = 1 if self.data_format == 'NCHW' else -1

        print('[INFO] :: Reading data from ->', source_dir)

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
                
                # process images
                img = np.array(file[data_name]['train']['img'])[idx]
                img = switch_dim(img)
                img_.append(img)
                del img
                gc.collect()
                
                lumen = np.array(file[data_name]['train']['lumen'])[idx]
                lumen = switch_dim(lumen).astype(np.uint8)
                media = np.array(file[data_name]['train']['media'])[idx]
                media = switch_dim(media).astype(np.uint8)
                mask_.append(lumen + media)

                del lumen
                del media
                gc.collect()

        self.input = np.expand_dims(np.vstack(img_), expanded_dim).astype(np.float32) / 255
        del img_
        gc.collect()
        self.y = np.vstack(mask_)
        del mask_
        gc.collect()

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
