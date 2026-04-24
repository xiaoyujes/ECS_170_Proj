'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting
import numpy as np


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        print('loading dataset and preparing split...')

        loaded_data = self.dataset.load()

        X_train = np.asarray(loaded_data['X_train'], dtype=np.float32)
        y_train = np.asarray(loaded_data['y_train'], dtype=np.int64)

        X_test = np.asarray(loaded_data['X_test'], dtype=np.float32)
        y_test = np.asarray(loaded_data['y_test'], dtype=np.int64)

        assert X_train.ndim == 2 and X_test.ndim == 2
        assert y_train.ndim == 1 and y_test.ndim == 1
        assert X_train.shape[1] == X_test.shape[1]

        self.method.data = {
            'train': {
                'X': X_train,
                'y': y_train
            },
            'test': {
                'X': X_test,
                'y': y_test
            }
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return score, learned_result