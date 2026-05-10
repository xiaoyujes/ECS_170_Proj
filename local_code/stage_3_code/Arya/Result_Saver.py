'''
Concrete result saver class.
Shared across MNIST, ORL, and CIFAR experiments.
'''

# Copyright (c) 2015-Present, ECS 189G
# All rights reserved.

import json
import os


class Result_Saver:
    """
    Saves evaluation results and prediction arrays to disk.

    Attributes
    ----------
    result_destination_folder_path : str  – output directory
    result_destination_file_name   : str  – base filename (no extension)
    data : dict                           – results to save
    """

    def __init__(self,
                 folder_path='../../result/stage_3_result',
                 file_name='result'):
        self.result_destination_folder_path = folder_path
        self.result_destination_file_name   = file_name
        self.data = None

    def save(self):
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        out_path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name + '.json'
        )

        # Convert numpy types to native Python for JSON serialisation
        def _convert(obj):
            if hasattr(obj, 'item'):        # numpy scalar
                return obj.item()
            if hasattr(obj, 'tolist'):      # numpy array
                return obj.tolist()
            return obj

        serialisable = {}
        for k, v in self.data.items():
            try:
                serialisable[k] = _convert(v)
            except Exception:
                serialisable[k] = str(v)

        with open(out_path, 'w') as f:
            json.dump(serialisable, f, indent=2)

        print(f'  [Result_Saver] Saved → {out_path}')
        return out_path
