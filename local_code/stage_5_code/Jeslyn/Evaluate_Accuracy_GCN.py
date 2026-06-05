'''
Style taken from Evaluate_Accuracy.py
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
import torch
import numpy as np

class Evaluate_Accuracy_GCN(evaluate):
    data = None

    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        #conversion to numpy if format is torch tensor
        if torch.is_tensor(true_y):
            true_y = true_y.numpy()
        if torch.is_tensor(pred_y):
            pred_y = pred_y.numpy()

        #if format is list, convert to numpy
        if isinstance(true_y, list):
            true_y = np.array(true_y)
        if isinstance(pred_y, list):
            pred_y = np.array(pred_y)

        #compute acc
        accuracy = accuracy_score(true_y, pred_y)

        return accuracy