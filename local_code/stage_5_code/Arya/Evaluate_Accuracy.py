'''
Concrete evaluate class for Accuracy metric.
Shared across MNIST, ORL, and CIFAR experiments.
'''

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Evaluate_Accuracy:

    def __init__(self):
        self.data = None  # {'true_y': ..., 'pred_y': ...}

    def evaluate(self):
        true_y = np.asarray(self.data['true_y'])
        pred_y = np.asarray(self.data['pred_y'])

        accuracy = float((true_y == pred_y).mean())

        metrics = {}
        for avg in ['weighted', 'macro', 'micro']:
            metrics[avg] = {
                'f1':        f1_score(true_y, pred_y, average=avg, zero_division=0),
                'precision': precision_score(true_y, pred_y, average=avg, zero_division=0),
                'recall':    recall_score(true_y, pred_y, average=avg, zero_division=0),
            }

        return {
            'accuracy': accuracy,
            'metrics':  metrics,
            'n_samples': len(true_y),
        }

    @staticmethod
    def result_to_str(result):
        lines = [
            '************ Overall Performance ************',
            f"CNN Accuracy: {result['accuracy']:.4f}",
        ]
        for avg, vals in result['metrics'].items():
            lines.append(f"F1-Score {avg}: {vals['f1']}")
            lines.append(f"Recall {avg}: {vals['recall']}")
            lines.append(f"Precision {avg}: {vals['precision']}")
        lines.append('************ Finish ************')
        return '\n'.join(lines)