'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

from local_code.base_class.result import result
import pickle
import os


class Result_Saver(result):

    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):

        print('saving results...')

        os.makedirs(
            self.result_destination_folder_path,
            exist_ok=True
        )

        file_path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name
            + '_'
            + str(self.fold_count)
        )

        save_data = {
            "pred_y":
                self.data["pred_y"],

            "true_y":
                self.data["true_y"],

            "history":
                self.data["history"],

            "final_test_loss":
                self.data["final_test_loss"]
        }

        with open(file_path, 'wb') as f:

            pickle.dump(
                save_data,
                f
            )