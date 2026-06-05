'''
Concrete GCN method class for CiteSeer node classification.
Graph Convolutional Network (Kipf & Welling, 2017).
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# ---------------------------------------------------------------------------
# GCN Layer
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """Single GCN layer: H' = AHW"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output  = torch.spmm(adj, support)
        return output + self.bias


# ---------------------------------------------------------------------------
# GCN Model
# ---------------------------------------------------------------------------

class GCN_Net(nn.Module):
    """
    Two-layer GCN:
    Input -> GCNLayer(in, hidden) -> ReLU -> Dropout
          -> GCNLayer(hidden, num_classes) -> LogSoftmax
    """

    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gc1     = GraphConvolution(in_features, hidden_dim)
        self.gc2     = GraphConvolution(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class Method_GCN:

    def __init__(self,
                 in_features,
                 hidden_dim=64,
                 num_classes=6,
                 dropout=0.5,
                 lr=1e-2,
                 weight_decay=5e-4,
                 max_epoch=200,
                 device=None):

        self.max_epoch = max_epoch
        self.device    = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GCN_Net(in_features, hidden_dim, num_classes, dropout).to(self.device)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

        self.train_loss_history = []
        self.train_acc_history  = []
        self.test_loss_history  = []
        self.test_acc_history   = []

    def _accuracy(self, logits, labels, idx):
        preds   = logits[idx].argmax(dim=1)
        correct = (preds == labels[idx]).sum().item()
        return correct / len(idx)

    def train(self, graph, train_test_val):
        features  = graph['X'].to(self.device)
        labels    = graph['y'].to(self.device)
        adj       = graph['utility']['A'].to(self.device)
        idx_train = train_test_val['idx_train'].to(self.device)
        idx_test  = train_test_val['idx_test'].to(self.device)

        print(f'\n[Method_GCN] Training on {self.device}')
        print(f'  Nodes={features.shape[0]}, Features={features.shape[1]}, Classes={labels.max().item()+1}')
        print(f'  Train={len(idx_train)}, Test={len(idx_test)}, Epochs={self.max_epoch}\n')

        for epoch in range(1, self.max_epoch + 1):
            t0 = time.time()

            # train
            self.model.train()
            self.optimizer.zero_grad()
            logits   = self.model(features, adj)
            tr_loss  = self.criterion(logits[idx_train], labels[idx_train])
            tr_loss.backward()
            self.optimizer.step()
            tr_acc = self._accuracy(logits, labels, idx_train)

            # test
            self.model.eval()
            with torch.no_grad():
                logits  = self.model(features, adj)
                te_loss = self.criterion(logits[idx_test], labels[idx_test]).item()
                te_acc  = self._accuracy(logits, labels, idx_test)

            self.train_loss_history.append(tr_loss.item())
            self.train_acc_history.append(tr_acc)
            self.test_loss_history.append(te_loss)
            self.test_acc_history.append(te_acc)

            if epoch % 20 == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f'  Epoch [{epoch:>3}/{self.max_epoch}] '
                      f'| Train Loss: {tr_loss.item():.4f}  Acc: {tr_acc*100:.2f}% '
                      f'| Test  Loss: {te_loss:.4f}  Acc: {te_acc*100:.2f}% '
                      f'| {elapsed:.2f}s')

        print(f'\n  Final Test Accuracy: {self.test_acc_history[-1]*100:.2f}%')

    def predict(self, graph, idx):
        features = graph['X'].to(self.device)
        adj      = graph['utility']['A'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features, adj)
        return logits[idx].argmax(dim=1).cpu().numpy()