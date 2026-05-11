from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class Method_Cifar10(method, nn.Module):

    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 64
    patience = 5

    def __init__(self, mName="CNN", mDescription=""):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.in_channels = 1
        self.num_classes = 10

        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, self.num_classes)

    def set_in_channels(self, in_channels):
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x

    def fit(self, train_loader, test_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        best_f1 = 0
        patience_counter = 0
        best_model_state = None

        self.history = {
            "epoch": [],
            "loss": [],
            "acc": [],
            "f1": []
        }

        for epoch in range(self.max_epoch):

            self.train()
            total_loss = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                logits = self.forward(images)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            self.eval()

            y_true, y_pred = [], []

            with torch.no_grad():
                for images, labels in test_loader:

                    images = images.to(device)

                    logits = self.forward(images)
                    preds = torch.argmax(logits, dim=1)

                    y_true.extend(labels.numpy())
                    y_pred.extend(preds.cpu().numpy())

            y_true = np.array(y_true).reshape(-1)
            y_pred = np.array(y_pred).reshape(-1)

            acc = np.mean(y_true == y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            self.history["epoch"].append(epoch)
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(acc)
            self.history["f1"].append(f1)

            print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def run(self):

        train_loader = self.data["train_loader"]
        test_loader = self.data["test_loader"]

        self.fit(train_loader, test_loader)

        device = next(self.parameters()).device
        loss_fn = nn.CrossEntropyLoss()

        y_true, y_pred = [], []
        total_loss = 0

        self.eval()

        with torch.no_grad():
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device).long()

                logits = self.forward(images)
                loss = loss_fn(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        final_loss = total_loss / len(test_loader)

        return {
            "pred_y": np.array(y_pred).reshape(-1),
            "true_y": np.array(y_true).reshape(-1),
            "history": self.history,
            "final_test_loss": final_loss
        }