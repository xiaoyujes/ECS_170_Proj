'''
Concrete IO class for ORL face dataset.


Data format inside the pickle:
    {
        'train': [ {'image': ndarray(112,92,3), 'label': int}, ... ],
        'test':  [ {'image': ndarray(112,92,3), 'label': int}, ... ]
    }

Since the image is grayscale stored as 3 identical RGB channels,
we take only the R channel → shape becomes (1, 112, 92).
Labels are integers from {1, 2, ..., 40}.
'''

import pickle
import numpy as np


class Dataset_Loader_ORL:

    def __init__(self):
        self.data_root = '../../data/stage_3_data/ORL'

    def load(self):
        with open(self.data_root, 'rb') as f:
            raw = pickle.load(f)

        def extract(split):
            X, y = [], []
            for instance in raw[split]:
                img = np.array(instance['image'], dtype=np.float32) / 255.0
                # (112, 92, 3) -> take R channel -> (112, 92) -> (1, 112, 92)
                X.append(img[:, :, 0][np.newaxis, :, :])
                y.append(int(instance['label']) - 1)  # shift {1..40} -> {0..39}
            return np.stack(X), np.array(y, dtype=np.int64)

        X_train, y_train = extract('train')
        X_test,  y_test  = extract('test')

        print(f'[ORL] Train: {X_train.shape}, Test: {X_test.shape}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
        }
