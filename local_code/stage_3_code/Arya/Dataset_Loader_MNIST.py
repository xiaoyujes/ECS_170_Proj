'''
Concrete IO class for MNIST dataset.
Loads from the instructor-provided pickle file named 'MNIST' (no extension).

Data format inside the pickle:
    {
        'train': [ {'image': ndarray(28,28), 'label': int}, ... ],
        'test':  [ {'image': ndarray(28,28), 'label': int}, ... ]
    }
'''

import pickle
import numpy as np


class Dataset_Loader_MNIST:

    def __init__(self):
        self.data_root = '../../data/stage_3_data/MNIST'

    def load(self):
        file_path = self.data_root  # full path to the pickle file named 'MNIST'

        with open(file_path, 'rb') as f:
            raw = pickle.load(f)

        def extract(split):
            X, y = [], []
            for instance in raw[split]:
                img = np.array(instance['image'], dtype=np.float32) / 255.0
                # shape: (28,28) -> add channel dim -> (1,28,28)
                X.append(img[np.newaxis, :, :])
                y.append(int(instance['label']))
            return np.stack(X), np.array(y, dtype=np.int64)

        X_train, y_train = extract('train')
        X_test,  y_test  = extract('test')

        print(f'[MNIST] Train: {X_train.shape}, Test: {X_test.shape}')

        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
        }
