'''
Source code: https://github.com/tkipf/pygcn
'''

from local_code.base_class.method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_2Layer(nn.Module):
    """Two-layer Graph Convolutional Network"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_2Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN_3Layer(nn.Module):
    """Three-layer Graph Convolutional Network"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_3Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


class Method_GCN(method):
    data = None
    max_epoch = 300
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_size = 16
    dropout_rate = 0.5
    patience = 20
    num_layers = 2

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        self.net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.historical_train_loss = []
        self.historical_test_loss = []
        self.historical_train_acc = []
        self.historical_val_acc = []
        self.best_val_acc = 0
        self.best_val_loss = float('inf')  # FIXED: Added this line
        self.best_test_loss = float('inf')
        self.patience_counter = 0

    def build_model(self, input_dim, output_dim):
        """Initialize GCN model based on num_layers"""
        print(f"Using {self.num_layers}-layer GCN")
        if self.num_layers == 3:
            self.net = GCN_3Layer(
                nfeat=input_dim,
                nhid=self.hidden_size,
                nclass=output_dim,
                dropout=self.dropout_rate
            )
        else:
            self.net = GCN_2Layer(
                nfeat=input_dim,
                nhid=self.hidden_size,
                nclass=output_dim,
                dropout=self.dropout_rate
            )
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def train_epoch(self, features, adj, labels, train_nodes):
        """Train for one epoch"""
        self.net.train()
        self.optimizer.zero_grad()

        output = self.net(features, adj)
        loss = self.criterion(output[train_nodes], labels[train_nodes])
        loss.backward()
        self.optimizer.step()

        # Calculate training accuracy
        _, pred = output[train_nodes].max(1)
        correct = pred.eq(labels[train_nodes]).sum().item()
        acc = correct / len(train_nodes)

        return loss.item(), acc

    def evaluate(self, features, adj, labels, nodes):
        """Evaluate on given nodes"""
        self.net.eval()
        with torch.no_grad():
            output = self.net(features, adj)
            loss = self.criterion(output[nodes], labels[nodes])
            _, pred = output[nodes].max(1)
            correct = pred.eq(labels[nodes]).sum().item()
            acc = correct / len(nodes)
        return loss.item(), acc

    def train(self, graph, train_nodes, val_nodes, test_nodes):
        """Train the GCN model with test loss tracking"""
        features = graph['X']
        adj = graph['utility']['A']
        labels = graph['y']

        input_dim = features.shape[1]
        output_dim = labels.max().item() + 1
        self.build_model(input_dim, output_dim)

        print(f'Training nodes: {len(train_nodes)}, Validation nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}')
        print(f'Model: {input_dim} features -> {self.hidden_size} hidden -> {output_dim} classes')

        for epoch in range(self.max_epoch):
            # Training
            train_loss, train_acc = self.train_epoch(features, adj, labels, train_nodes)

            # Validation (for early stopping)
            val_loss, val_acc = self.evaluate(features, adj, labels, val_nodes)

            # Test loss tracking (for plotting)
            test_loss, test_acc = self.evaluate(features, adj, labels, test_nodes)

            # Store history
            self.historical_train_loss.append(train_loss)
            self.historical_test_loss.append(test_loss)
            self.historical_train_acc.append(train_acc)
            self.historical_val_acc.append(val_acc)

            # Print progress
            if epoch % 20 == 0:
                print(f'Epoch {epoch:3d}/{self.max_epoch} | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                      f'Test Loss: {test_loss:.4f}')

            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.best_model_state = self.net.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch} (no improvement in validation loss)')
                    break

        # Restore best model
        self.net.load_state_dict(self.best_model_state)
        print(f'Finished Training. Best Validation Accuracy: {self.best_val_acc:.4f}, Best Validation Loss: {self.best_val_loss:.4f}')

    def test(self, graph, test_nodes):
        """Test the trained model"""
        self.net.eval()
        features = graph['X']
        adj = graph['utility']['A']

        with torch.no_grad():
            output = self.net(features, adj)
            test_loss = self.criterion(output[test_nodes], graph['y'][test_nodes])
            _, predictions = output[test_nodes].max(1)

        return predictions.numpy(), test_loss.item()

    def run(self):
        """Main method to run training and testing"""
        print('GCN method running...')

        graph = self.data['graph']
        train_nodes = self.data['train_test_val']['idx_train']
        val_nodes = self.data['train_test_val']['idx_val']
        test_nodes = self.data['train_test_val']['idx_test']

        print('--start training...')
        self.train(graph, train_nodes, val_nodes, test_nodes)

        print('--start testing...')
        pred_y, test_loss = self.test(graph, test_nodes)
        true_y = graph['y'][test_nodes].numpy()

        correct = (pred_y == true_y).sum()
        total = len(true_y)
        accuracy = 100 * correct // total
        print(f'Accuracy on {total} test nodes: {accuracy}%')
        print(f'Test Loss: {test_loss:.4f}')

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'train_loss_history': self.historical_train_loss,
            'test_loss_history': self.historical_test_loss,
            'train_acc_history': self.historical_train_acc,
            'val_acc_history': self.historical_val_acc,
            'test_loss': test_loss
        }