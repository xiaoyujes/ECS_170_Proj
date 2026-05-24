'''
Source Code: 'Result_Saver.py' from '/stage_1_code'
'''

from local_code.base_class.result import result
import pickle
import os

class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        if 'generated_text' in self.data:
            txt_path = self.result_destination_folder_path + self.result_destination_file_name + 'text.txt'

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('=== Generated Text Samples ===\n\n')
                for i, sample in enumerate(self.data['generated_text']):
                    f.write(f'Sample {i + 1}:\n')
                    f.write(f"  Starting words : {sample['starting_words']}\n")
                    f.write(f"  Generated text : {sample['generated']}\n\n")
            print(f'Generated text saved to: {txt_path}')

            pkl_path = self.result_destination_folder_path + self.result_destination_file_name + 'results.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f'Full results pickled to: {pkl_path}')