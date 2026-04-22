# Arya Chaudhari Made this Model

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 300   # good balance for your laptop
    learning_rate = 6e-4

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # --- Architecture (same as your working version) ---
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 10)


        self.historical_loss = []

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)   # logits (correct for CrossEntropyLoss)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # FULL BATCH (required by assignment)
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):

            preds = self.forward(X)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            self.historical_loss.append(loss.item())

            if epoch % 10 == 0:
                acc = (preds.argmax(1) == y).float().mean().item()
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | Accuracy {acc:.4f}")

    def test(self, X):
        X = torch.FloatTensor(np.array(X))

        with torch.no_grad():
            preds = self.forward(X)

        # IMPORTANT: convert tensor → numpy (fixes sklearn + printing issues)
        return preds.argmax(1).cpu().numpy()

    def run(self):
        print("method running...")
        print("--start training...")
        self.train(self.data['train']['X'], self.data['train']['y'])

        print("--start testing...")
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': np.array(self.data['test']['y'])
        }