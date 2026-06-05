from local_code.base_class.method import method
import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """One graph convolution layer for A_hat X W."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class Method_GCN(method, nn.Module):
    data = None

    max_epoch = 200
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 16
    hidden_dims = None
    dropout = 0.5
    use_best_validation_model = True
    best_selection_metric = 'accuracy'
    print_interval = 10
    early_stopping_patience = None
    use_validation = True

    def __init__(self, mName='graph convolutional network', mDescription=''):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.layers = None
        self.history = []
        self.best_epoch = None
        self.best_validation_score = None
        self.best_model_state = None

    def build_model(self, num_features, num_classes):
        hidden_dims = self.hidden_dims if self.hidden_dims else [self.hidden_dim]
        layer_dims = [num_features] + list(hidden_dims) + [num_classes]
        self.layers = nn.ModuleList([
            GraphConvolution(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)

    def accuracy(self, output, labels):
        pred_y = output.max(1)[1].type_as(labels)
        correct = pred_y.eq(labels).double().sum()
        return correct / len(labels)

    def train_model(self, features, labels, adj, idx_train, idx_val):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        patience_count = 0

        for epoch in range(self.max_epoch):
            t = time.time()
            self.train()
            optimizer.zero_grad()

            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = self.accuracy(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            if self.use_validation and idx_val is not None and len(idx_val) > 0:
                self.eval()
                with torch.no_grad():
                    output = self.forward(features, adj)
                    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                    acc_val = self.accuracy(output[idx_val], labels[idx_val])
                loss_val_item = loss_val.item()
                acc_val_item = acc_val.item()
            else:
                loss_val_item = None
                acc_val_item = None

            row = {
                'epoch': epoch + 1,
                'loss_train': loss_train.item(),
                'accuracy_train': acc_train.item(),
                'loss_val': loss_val_item,
                'accuracy_val': acc_val_item,
                'time': time.time() - t
            }
            self.history.append(row)

            if self.use_validation and row['loss_val'] is not None:
                if self.best_selection_metric == 'loss':
                    validation_score = -row['loss_val']
                else:
                    validation_score = row['accuracy_val']

                if self.best_validation_score is None or validation_score > self.best_validation_score:
                    self.best_validation_score = validation_score
                    self.best_epoch = epoch + 1
                    self.best_model_state = copy.deepcopy(self.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

            if epoch % self.print_interval == 0 or epoch == self.max_epoch - 1:
                if row['loss_val'] is None:
                    print(
                        'Epoch: {:04d}'.format(epoch + 1),
                        'loss_train: {:.4f}'.format(row['loss_train']),
                        'acc_train: {:.4f}'.format(row['accuracy_train']),
                        'time: {:.4f}s'.format(row['time'])
                    )
                else:
                    print(
                        'Epoch: {:04d}'.format(epoch + 1),
                        'loss_train: {:.4f}'.format(row['loss_train']),
                        'acc_train: {:.4f}'.format(row['accuracy_train']),
                        'loss_val: {:.4f}'.format(row['loss_val']),
                        'acc_val: {:.4f}'.format(row['accuracy_val']),
                        'time: {:.4f}s'.format(row['time'])
                    )

            if self.use_validation and self.early_stopping_patience is not None and patience_count >= self.early_stopping_patience:
                print('Early stopping at epoch:', epoch + 1)
                break

    def test_model(self, features, labels, adj, idx_test):
        if self.use_best_validation_model and self.best_model_state is not None:
            print('Restoring best validation model from epoch:', self.best_epoch)
            self.load_state_dict(self.best_model_state)

        self.eval()
        with torch.no_grad():
            output = self.forward(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = self.accuracy(output[idx_test], labels[idx_test])
            pred_y = output[idx_test].max(1)[1]

        print('Test set results:',
              'loss= {:.4f}'.format(loss_test.item()),
              'accuracy= {:.4f}'.format(acc_test.item()))

        return {
            'pred_y': pred_y.cpu().numpy(),
            'true_y': labels[idx_test].cpu().numpy(),
            'test_loss': loss_test.item(),
            'test_accuracy': acc_test.item(),
            'best_epoch': self.best_epoch,
            'best_validation_score': self.best_validation_score,
            'history': self.history
        }

    def run(self):
        print('method running...')
        graph = self.data['graph']
        split = self.data['train_test_val']

        features = graph['X']
        labels = graph['y']
        adj = graph['utility']['A']
        idx_train = split['idx_train']
        idx_val = split['idx_val']
        idx_test = split['idx_test']

        self.build_model(features.shape[1], int(labels.max().item()) + 1)

        print('--start training...')
        self.train_model(features, labels, adj, idx_train, idx_val)

        print('--start testing...')
        return self.test_model(features, labels, adj, idx_test)
