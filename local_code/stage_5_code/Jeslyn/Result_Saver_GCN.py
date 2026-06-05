'''
Style taken from Result_Saver.py
'''

import csv
import json
from local_code.base_class.result import result

class Result_Saver_GCN(result):
    data = None
    dataset_name = None
    result_destination_folder_path = None
    result_destination_file_name = None
    fold_count = None

    def save(self):
        #saving predictions and evaluation metrics to files
        print('saving results for {} dataset...'.format(self.dataset_name))

        #saving predictions for node classification
        if 'predictions' in self.data:
            pred_file = self.result_destination_folder_path + self.result_destination_file_name + '_predictions_' + str(
                self.fold_count) + '.csv'

            with open(pred_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Node_Index', 'Predicted_Class'])

                for node_idx, pred in enumerate(self.data['predictions']):
                    writer.writerow([node_idx, pred])
            print('Predictions saved to: {}'.format(pred_file))

        #saving eval metrics
        if 'metrics' in self.data:
            metrics_file = self.result_destination_folder_path + self.result_destination_file_name + '_metrics_' + str(
                self.fold_count) + '.json'

            with open(metrics_file, 'w') as file:
                json.dump(self.data['metrics'], file, indent=4)
            print('Metrics saved to: {}'.format(metrics_file))

        #saving data (simple data --> list of predictions)
        elif not isinstance(self.data, dict):
            simple_file = self.result_destination_folder_path + self.result_destination_file_name + '_' + str(
                self.fold_count) + '.csv'

            with open(simple_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Prediction'])

                for pred in self.data:
                    writer.writerow([pred])
            print('Results saved to: {}'.format(simple_file))