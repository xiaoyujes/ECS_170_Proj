'''
Concrete RNN method class for IMDB sentiment classification.
Supports RNN, LSTM, and GRU via the unit_type parameter (for task 4-5).
Binary classification: pos=1, neg=0.
'''

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ---------------------------------------------------------------------------
# RNN architecture
# ---------------------------------------------------------------------------

class RNN_Classification_Net(nn.Module):
    """
    Embedding -> RNN/LSTM/GRU -> take last hidden state -> FC -> output

    Parameters
    ----------
    vocab_size  : int  – vocabulary size (including PAD and UNK)
    embed_dim   : int  – embedding dimension
    hidden_dim  : int  – RNN hidden dimension
    num_layers  : int  – number of stacked RNN layers
    num_classes : int  – output classes (2 for binary)
    dropout     : float
    unit_type   : str  – 'RNN' | 'LSTM' | 'GRU'
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.5,
                 unit_type='RNN'):
        super().__init__()

        self.unit_type  = unit_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # padding_idx=0 so PAD tokens don't contribute to gradients
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if unit_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=rnn_dropout)
        elif unit_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=rnn_dropout)
        else:  # plain RNN
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=rnn_dropout,
                              nonlinearity='tanh')

        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch, seq, embed)

        if self.unit_type == 'LSTM':
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)

        # hidden: (num_layers, batch, hidden_dim) — take top layer
        out = self.dropout(hidden[-1])               # (batch, hidden_dim)
        return self.fc(out)                          # (batch, num_classes)


# ---------------------------------------------------------------------------
# Training / evaluation wrapper
# ---------------------------------------------------------------------------

class Method_RNN_Classification:

    def __init__(self,
                 vocab_size=10002,
                 embed_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.5,
                 unit_type='RNN',
                 lr=1e-3,
                 batch_size=128,
                 max_epoch=10,
                 device=None):

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = RNN_Classification_Net(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            unit_type=unit_type,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=5, gamma=0.5)

        self.train_loss_history = []
        self.train_acc_history  = []
        self.test_loss_history  = []
        self.test_acc_history   = []

    def _make_loader(self, X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=0)

    def _run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * y_batch.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == y_batch).sum().item()
                total      += y_batch.size(0)

        return total_loss / total, correct / total

    def train(self, X_train, y_train, X_test, y_test):
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        test_loader  = self._make_loader(X_test,  y_test,  shuffle=False)

        print(f'\n[Method_RNN_Classification] Training on {self.device}')
        print(f'  Samples - train: {len(y_train)}, test: {len(y_test)}')
        print(f'  Epochs={self.max_epoch}, Batch={self.batch_size}\n')

        for epoch in range(1, self.max_epoch + 1):
            t0 = time.time()

            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            te_loss, te_acc = self._run_epoch(test_loader,  train=False)
            self.scheduler.step()

            self.train_loss_history.append(tr_loss)
            self.train_acc_history.append(tr_acc)
            self.test_loss_history.append(te_loss)
            self.test_acc_history.append(te_acc)

            elapsed = time.time() - t0
            print(f'  Epoch [{epoch:>3}/{self.max_epoch}] '
                  f'| Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% '
                  f'| Test  Loss: {te_loss:.4f}  Acc: {te_acc*100:.2f}% '
                  f'| {elapsed:.1f}s')

        print(f'\n  Final Test Accuracy: {self.test_acc_history[-1]*100:.2f}%')

    def predict(self, X):
        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.int64),
                                   shuffle=False)
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits  = self.model(X_batch)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)
