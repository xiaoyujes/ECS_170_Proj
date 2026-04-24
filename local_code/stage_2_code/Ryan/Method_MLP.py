from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-3
    batch_size = 128
    patience = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def fit(self, X, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
        y_train = torch.tensor(np.array(y_train), dtype=torch.long).to(device)
        X_val = torch.tensor(np.array(X_val), dtype=torch.float32).to(device)
        y_val = torch.tensor(np.array(y_val), dtype=torch.long).to(device)

        self.to(device)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        loss_fn = nn.CrossEntropyLoss()

        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None

        self.history = {
            "epoch": [],
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        for epoch in range(0, self.max_epoch, 10):

            self.train()
            total_loss = 0
            num_batches = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            self.eval()
            with torch.no_grad():
                logits = self.forward(X_val)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                y_true = y_val.cpu().numpy()
                y_pred = preds.cpu().numpy()

                acc = (y_pred == y_true).mean()
                precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            self.history["epoch"].append(epoch)
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(acc)
            self.history["precision"].append(precision)
            self.history["recall"].append(recall)
            self.history["f1"].append(f1)

            print(
                f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                f"Val Acc: {acc:.4f} | P: {precision:.4f} | "
                f"R: {recall:.4f} | F1: {f1:.4f}"
            )

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def test(self, X):
        device = next(self.parameters()).device
        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        return preds.cpu().numpy()

    def run(self):
        self.fit(self.data["train"]["X"], self.data["train"]["y"])

        device = next(self.parameters()).device
        loss_fn = nn.CrossEntropyLoss()

        X_test = torch.tensor(np.array(self.data["test"]["X"]), dtype=torch.float32).to(device)
        y_test = torch.tensor(np.array(self.data["test"]["y"]), dtype=torch.long).to(device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(X_test)
            final_test_loss = loss_fn(logits, y_test).item()

        print("Final Test Loss:", final_test_loss)

        pred_y = self.test(self.data["test"]["X"])

        return {
            "pred_y": np.array(pred_y).reshape(-1),
            "true_y": np.array(self.data["test"]["y"]).reshape(-1),
            "history": self.history,
            "final_test_loss": final_test_loss
        }