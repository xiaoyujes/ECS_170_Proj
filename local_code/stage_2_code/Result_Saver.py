'''
Concrete ResultModule class for a specific experiment ResultModule output
'''
import csv

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        print('saving results...')
        f = self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count) + '.csv'

        with open(f, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Prediction'])

            for pred in self.data:
                writer.writerow([pred])