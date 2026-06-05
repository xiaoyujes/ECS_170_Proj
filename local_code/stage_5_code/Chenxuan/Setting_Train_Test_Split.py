from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        self.method.data = loaded_data
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        return metrics, learned_result
