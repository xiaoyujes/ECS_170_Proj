'''
Concrete IO class for CiteSeer node classification dataset.
Self-contained, no base class required.

Reads two files from the dataset folder:
    node  - tab separated: node_id, features..., label
    link  - tab separated: node_id_1, node_id_2
'''

import numpy as np
import scipy.sparse as sp
import torch


class Dataset_Loader_CiteSeer:

    def __init__(self):
        self.dataset_source_folder_path = '../../data/stage_5_data/citeseer'

    def adj_normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv  = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).dot(r_mat_inv)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices   = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values    = torch.from_numpy(sparse_mx.data)
        shape     = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes      = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    def load(self):
        print('Loading CiteSeer dataset...')

        idx_features_labels = np.genfromtxt(
            '{}/node'.format(self.dataset_source_folder_path), dtype=np.dtype(str))

        features       = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels  = self.encode_onehot(idx_features_labels[:, -1])

        idx         = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map     = {j: i for i, j in enumerate(idx)}
        reverse_idx = {i: j for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(
            '{}/link'.format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)

        # filter out None edges (nodes not in node file)
        valid = np.all(edges != None, axis=1)
        edges = edges[valid]

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(onehot_labels.shape[0], onehot_labels.shape[0]),
            dtype=np.float32)
        adj      = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        features = torch.FloatTensor(np.array(features.todense()))
        labels   = torch.LongTensor(np.where(onehot_labels)[1])
        adj      = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(range(120))
        idx_test  = torch.LongTensor(range(200, 1200))
        idx_val   = torch.LongTensor(range(1200, 1500))

        print(f'  Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {labels.max().item()+1}')
        print(f'  Train: {len(idx_train)}, Test: {len(idx_test)}, Val: {len(idx_val)}')

        graph = {
            'X': features,
            'y': labels,
            'utility': {'A': adj, 'reverse_idx': reverse_idx}
        }
        train_test_val = {
            'idx_train': idx_train,
            'idx_test':  idx_test,
            'idx_val':   idx_val,
        }
        return {'graph': graph, 'train_test_val': train_test_val}