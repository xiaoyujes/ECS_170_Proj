'''
Concrete evaluate class for Accuracy metric.
Shared across MNIST, ORL, and CIFAR experiments.
'''

# Copyright (c) 2015-Present, ECS 189G
# All rights reserved.

import numpy as np


class Evaluate_Accuracy:
    """
    Computes accuracy and a simple per-class breakdown.

    Usage
    -----
    evaluator = Evaluate_Accuracy()
    evaluator.data = {'true_y': y_true, 'pred_y': y_pred}
    result = evaluator.evaluate()
    print(evaluator.result_to_str(result))
    """

    def __init__(self):
        self.data = None   # set externally: {'true_y': ..., 'pred_y': ...}

    def evaluate(self):
        true_y = np.asarray(self.data['true_y'])
        pred_y = np.asarray(self.data['pred_y'])

        if true_y.shape != pred_y.shape:
            raise ValueError('Shape mismatch between true_y and pred_y.')

        overall_acc = float((true_y == pred_y).mean())

        classes = np.unique(true_y)
        per_class = {}
        for c in classes:
            mask = (true_y == c)
            per_class[int(c)] = float((pred_y[mask] == c).mean())

        return {
            'accuracy':  overall_acc,
            'per_class': per_class,
            'n_samples': len(true_y),
        }

    @staticmethod
    def result_to_str(result):
        lines = [
            '=' * 45,
            f"  Overall Accuracy : {result['accuracy']*100:.2f}%",
            f"  Total Samples    : {result['n_samples']}",
            '-' * 45,
            '  Per-Class Accuracy:',
        ]
        for cls, acc in sorted(result['per_class'].items()):
            lines.append(f'    Class {cls:>3} : {acc*100:.2f}%')
        lines.append('=' * 45)
        return '\n'.join(lines)
