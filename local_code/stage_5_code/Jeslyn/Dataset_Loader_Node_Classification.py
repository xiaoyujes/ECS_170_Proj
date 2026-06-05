'''
Concrete IO class for a specific dataset with random stratified sampling
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
import random

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)
        # Convert seed to integer if it's provided as string or None
        if seed is not None:
            try:
                self.seed = int(seed)
            except (ValueError, TypeError):
                self.seed = 42
        else:
            self.seed = 42

        # Create a dedicated random generator for this instance
        self.rng = np.random.RandomState(self.seed)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def stratified_split(self, labels, train_per_class, test_per_class, val_ratio=0.1):
        """
        Perform stratified sampling for train/test/val splits

        Args:
            labels: numpy array of labels
            train_per_class: number of training samples per class
            test_per_class: number of test samples per class
            val_ratio: ratio of validation from remaining data
        """
        # Use instance-specific random generator (doesn't affect global state)
        unique_classes = np.unique(labels)
        train_idx = []
        test_idx = []
        val_idx = []

        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            # Use instance rng instead of global np.random
            shuffled_indices = self.rng.permutation(cls_indices)

            # Take train_per_class for training
            if len(shuffled_indices) >= train_per_class:
                train_idx.extend(shuffled_indices[:train_per_class])
                remaining = shuffled_indices[train_per_class:]
            else:
                # If not enough samples, take all
                train_idx.extend(shuffled_indices)
                remaining = np.array([])

            # Take test_per_class for testing
            if len(remaining) >= test_per_class:
                test_idx.extend(remaining[:test_per_class])
                remaining = remaining[test_per_class:]
            elif len(remaining) > 0:
                test_idx.extend(remaining)
                remaining = np.array([])

            # Remaining for validation
            if len(remaining) > 0:
                val_idx.extend(remaining)

        # Shuffle and limit validation size
        val_idx = np.array(val_idx)
        if len(val_idx) > 500:
            val_idx = self.rng.permutation(val_idx)
            val_idx = val_idx[:500]

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    def load(self):
        """Load citation network dataset with random stratified splits"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])
        labels_np = np.where(onehot_labels)[1]  # Get class indices

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels_np)
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # Create stratified splits based on project requirements
        if self.dataset_name == 'cora':
            # 7 classes, 20 per class for training = 140, 150 per class for test = 1050
            train_idx, val_idx, test_idx = self.stratified_split(
                labels_np, train_per_class=20, test_per_class=150, val_ratio=0.1
            )
            print(f'\nCora dataset statistics:')
            print(f'  Total nodes: {len(labels_np)}')
            print(f'  Training nodes: {len(train_idx)} (20 per class x 7 classes)')
            print(f'  Test nodes: {len(test_idx)} (150 per class x 7 classes)')
            print(f'  Validation nodes: {len(val_idx)}')

        elif self.dataset_name == 'citeseer':
            # 6 classes, 20 per class for training = 120, 200 per class for test = 1200
            train_idx, val_idx, test_idx = self.stratified_split(
                labels_np, train_per_class=20, test_per_class=200, val_ratio=0.1
            )
            print(f'\nCiteseer dataset statistics:')
            print(f'  Total nodes: {len(labels_np)}')
            print(f'  Training nodes: {len(train_idx)} (20 per class x 6 classes)')
            print(f'  Test nodes: {len(test_idx)} (200 per class x 6 classes)')
            print(f'  Validation nodes: {len(val_idx)}')

        elif self.dataset_name == 'pubmed':
            # 3 classes, 20 per class for training = 60, 200 per class for test = 600
            train_idx, val_idx, test_idx = self.stratified_split(
                labels_np, train_per_class=20, test_per_class=200, val_ratio=0.1
            )
            print(f'\nPubmed dataset statistics:')
            print(f'  Total nodes: {len(labels_np)}')
            print(f'  Training nodes: {len(train_idx)} (20 per class x 3 classes)')
            print(f'  Test nodes: {len(test_idx)} (200 per class x 3 classes)')
            print(f'  Validation nodes: {len(val_idx)}')

        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            train_idx = np.array(range(5))
            val_idx = np.array(range(5, 10))
            test_idx = np.array(range(5, 10))

        idx_train = torch.LongTensor(train_idx)
        idx_val = torch.LongTensor(val_idx)
        idx_test = torch.LongTensor(test_idx)

        # Print class distribution for verification
        print(f'\nClass distribution in splits for {self.dataset_name}:')
        unique_classes = np.unique(labels_np)
        print(f"{'Class':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-" * 32)
        for class_idx in unique_classes:
            train_count = np.sum(labels_np[train_idx] == class_idx)
            val_count = np.sum(labels_np[val_idx] == class_idx)
            test_count = np.sum(labels_np[test_idx] == class_idx)
            print(f"{class_idx:<8} {train_count:<8} {val_count:<8} {test_count:<8}")

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}