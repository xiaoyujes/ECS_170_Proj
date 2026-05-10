'''
Concrete CNN method class for MNIST digit classification.
'''

# Copyright (c) 2015-Present, ECS 189G
# All rights reserved.

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

class CNN_MNIST_Net(nn.Module):
    """
    A small CNN suited for 28×28 grey-scale digit images.

    Architecture
    ============
    Conv(1→32, 3×3, pad=1)  → BN → ReLU → MaxPool(2×2)   [14×14]
    Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   [7×7]
    Conv(64→128, 3×3, pad=1)→ BN → ReLU                   [7×7]
    Flatten → FC(128*7*7 → 256) → ReLU → Dropout(0.5)
    FC(256 → 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14 → 7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training / evaluation wrapper
# ---------------------------------------------------------------------------

class Method_CNN_MNIST:
    """
    Wraps CNN_MNIST_Net with training, evaluation, and plotting logic.

    Parameters
    ----------
    num_classes   : int   – number of output classes (default 10)
    lr            : float – learning rate            (default 1e-3)
    batch_size    : int   – mini-batch size          (default 64)
    max_epoch     : int   – training epochs          (default 30)
    device        : str   – 'cuda' | 'cpu'          (auto-detected)
    """

    def __init__(self,
                 num_classes=10,
                 lr=1e-3,
                 batch_size=128,
                 max_epoch=10,
                 device=None):

        self.num_classes = num_classes
        self.lr          = lr
        self.batch_size  = batch_size
        self.max_epoch   = max_epoch
        self.device      = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model     = CNN_MNIST_Net(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=10, gamma=0.5)

        # history
        self.train_loss_history = []
        self.train_acc_history  = []
        self.test_loss_history  = []
        self.test_acc_history   = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.float32)
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
                    self.optimizer.step()

                total_loss += loss.item() * y_batch.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == y_batch).sum().item()
                total      += y_batch.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, X_test, y_test):
        """Train for max_epoch epochs and record learning curves."""
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        test_loader  = self._make_loader(X_test,  y_test,  shuffle=False)

        print(f'\n[Method_CNN_MNIST] Training on {self.device}')
        print(f'  Samples  – train: {len(y_train)}, test: {len(y_test)}')
        print(f'  Epochs={self.max_epoch}, LR={self.lr}, Batch={self.batch_size}\n')

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
        """Return predicted class indices for input array X."""
        self.model.eval()
        loader = self._make_loader(X,
                                   np.zeros(len(X), dtype=np.int64),
                                   shuffle=False)
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits  = self.model(X_batch)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)
