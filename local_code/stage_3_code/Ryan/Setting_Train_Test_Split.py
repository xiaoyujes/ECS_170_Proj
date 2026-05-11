'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        print('loading dataset and preparing split...')

        loaded_data = self.dataset.load()

        train_loader = loaded_data[
            'train_loader'
        ]

        test_loader = loaded_data[
            'test_loader'
        ]

        self.method.data = {
            'train_loader':
                train_loader,

            'test_loader':
                test_loader
        }

        learned_result = self.method.run()

        self.result.data = learned_result

        self.result.save()

        self.evaluate.data = learned_result

        score = self.evaluate.evaluate()

        return score, learned_result