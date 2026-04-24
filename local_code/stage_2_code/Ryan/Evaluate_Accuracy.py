'''
Concrete Evaluate class for a specific evaluation metrics
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
import numpy as np
import torch


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        return accuracy_score(y_true, y_pred)