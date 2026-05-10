'''
Concrete IO class for MNIST dataset
'''

# Copyright (c) 2015-Present, ECS 189G
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import struct
import numpy as np
import os


class Dataset_Loader_MNIST:
    """
    Loads the MNIST dataset from raw binary IDX files.

    Expected directory layout (relative to data_root):
        MNIST/
            train-images.idx3-ubyte
            train-labels.idx1-ubyte
            t10k-images.idx3-ubyte
            t10k-labels.idx1-ubyte

    Pixel values are normalised to [0, 1] and images are returned as
    (N, 1, 28, 28) float32 tensors so they can be fed directly into a
    PyTorch CNN (single-channel grey-scale).
    """

    def __init__(self, data_root='../../data/stage_3_data/MNIST'):
        self.data_root = data_root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_images(path):
        """Read an IDX3 image file and return a (N, 1, H, W) float32 array."""
        with open(path, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f'Invalid MNIST image file: {path}')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0
        return images

    @staticmethod
    def _read_labels(path):
        """Read an IDX1 label file and return a (N,) int64 array."""
        with open(path, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f'Invalid MNIST label file: {path}')
            labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        """
        Returns
        -------
        result : dict with keys
            'train' : {'X': ndarray (N,1,28,28), 'y': ndarray (N,)}
            'test'  : {'X': ndarray (M,1,28,28), 'y': ndarray (M,)}
        """
        train_img_path = os.path.join(self.data_root, 'train-images.idx3-ubyte')
        train_lbl_path = os.path.join(self.data_root, 'train-labels.idx1-ubyte')
        test_img_path  = os.path.join(self.data_root, 't10k-images.idx3-ubyte')
        test_lbl_path  = os.path.join(self.data_root, 't10k-labels.idx1-ubyte')

        for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'MNIST file not found: {p}')

        result = {
            'train': {
                'X': self._read_images(train_img_path),
                'y': self._read_labels(train_lbl_path),
            },
            'test': {
                'X': self._read_images(test_img_path),
                'y': self._read_labels(test_lbl_path),
            },
        }

        print(f'[MNIST] Train: {result["train"]["X"].shape}, '
              f'Test: {result["test"]["X"].shape}')
        return result
