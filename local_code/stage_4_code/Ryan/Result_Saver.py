from local_code.base_class.result import result
import pickle
import os


class Result_Saver(result):

    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):

        print('saving Stage 4 results...')

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
            "pred_y": self.data.get("pred_y", None),
            "true_y": self.data.get("true_y", None),
            "history": self.data.get("history", None),
            "final_test_loss": self.data.get("final_test_loss", None)
        }

        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)