'''
Concrete IO class for CIFAR-10 dataset.
Loads from the instructor-provided pickle file named 'CIFAR' (no extension).

Data format inside the pickle:
    {
        'train': [ {'image': ndarray(32,32,3), 'label': int}, ... ],
        'test':  [ {'image': ndarray(32,32,3), 'label': int}, ... ]
    }

Images are colored (3 channels), normalized to [0,1].
Output shape: (N, 3, 32, 32)
Labels: {0, 1, ..., 9}
'''

import pickle
import numpy as np


class Dataset_Loader_CIFAR:

    def __init__(self):
        self.data_root = '../../data/stage_3_data/CIFAR'

    def load(self):
        with open(self.data_root, 'rb') as f:
            raw = pickle.load(f)

        def extract(split):
            X, y = [], []
            for instance in raw[split]:
                img = np.array(instance['image'], dtype=np.float32) / 255.0
                # (32, 32, 3) -> (3, 32, 32)
                X.append(img.transpose(2, 0, 1))
                y.append(int(instance['label']))
            return np.stack(X), np.array(y, dtype=np.int64)

        X_train, y_train = extract('train')
        X_test,  y_test  = extract('test')

        print(f'[CIFAR] Train: {X_train.shape}, Test: {X_test.shape}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
        }
