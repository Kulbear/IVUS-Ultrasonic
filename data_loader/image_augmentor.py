from imgaug import augmenters as iaa

import numpy as np


class ImageAugmentor:
    def __init__(self, images, mode='SIMPLE', labels=None, end_to_end=True):
        self._source_images = images
        self._source_labels = labels
        self._mode = mode
        self._result_images = []
        self._result_labels = []
        self._end_to_end = end_to_end

    def generate(self):
        if self._mode == 'SIMPLE':
            self._geometry_augment()
        else:
            raise NotImplementedError

        if self._end_to_end:
            return np.vstack(self._result_images), np.vstack(
                self._result_labels)
        else:
            return self._result_images, self._result_labels

    def _geometry_augment(self, dropout_prob=0.15):
        # rotation
        rot_augs = [iaa.Affine(rotate=(deg)) for deg in range(0, 90, 15)]
        for aug in rot_augs:
            self._result_images.append(aug.augment_images(self._source_images))
            self._result_labels.append(aug.augment_images(self._source_labels))

        self._source_images = np.vstack(self._result_images.copy())
        self._source_labels = np.vstack(self._result_labels.copy())

        # for those need to modify the mask as well
        flip_lr = iaa.Fliplr(1)
        flip_ud = iaa.Flipud(1)
        flip_lrud = iaa.Sequential([flip_lr, flip_ud])
        mask_modified_augs = [flip_lr, flip_ud, flip_lrud]
        for aug in mask_modified_augs:
            self._result_images.append(aug.augment_images(self._source_images))
            self._result_labels.append(aug.augment_images(self._source_labels))

        # NOTE: directly convert all images to whole black without modify the mask
        # This is caused by the `imgaug` implementation
        # But we actually got benefited... by super strong regularization
        dropout = iaa.Dropout(p=dropout_prob)
        mask_no_modified_augs = [dropout]
        length = len(self._result_images)
        for aug in mask_no_modified_augs:
            for grp_idx in range(length):
                self._result_images.append(
                    aug.augment_images(
                        np.array((self._result_images[grp_idx])).astype(
                            np.uint8)))
                self._result_labels.append(self._result_labels[grp_idx])
