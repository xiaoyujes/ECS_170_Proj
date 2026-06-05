from local_code.base_class.result import result
import csv
import os
import pickle


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        pkl_path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name + '.pkl'
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.data, f)
        print('Prediction results saved to:', pkl_path)

        if self.data is not None and 'history' in self.data:
            history_path = os.path.join(
                self.result_destination_folder_path,
                self.result_destination_file_name + '_history.csv'
            )
            history = self.data['history']
            if history:
                with open(history_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)
                print('Training history saved to:', history_path)

    def load(self):
        pkl_path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name + '.pkl'
        )
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
