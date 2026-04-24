'''
Concrete IO class for a specific dataset
'''

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        import numpy as np
        import pandas as pd

        # build file paths
        train_path = self.dataset_source_folder_path + self.dataset_source_file_name[0]
        test_path = self.dataset_source_folder_path + self.dataset_source_file_name[1]

        # load CSVs
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)

        train_data = train_df.values
        test_data = test_df.values

        X_train = train_data[:, 1:].astype(np.float32)
        y_train = train_data[:, 0].astype(np.int64)

        X_test = test_data[:, 1:].astype(np.float32)
        y_test = test_data[:, 0].astype(np.int64)

        # normalize pixel values (0–255 → 0–1)
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        assert X_train.shape[1] == 784
        assert X_test.shape[1] == 784

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }