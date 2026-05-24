from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score


class Method_RNN(method, nn.Module):

    data = None

    max_epoch = 12
    learning_rate = 2e-3
    batch_size = 128
    patience = 3

    def __init__(self,
                 vocab_size,
                 embed_dim=100,
                 hidden_dim=96,
                 mName="RNN-STABLE",
                 mDescription=""):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = 2

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.6)

        self.fc = nn.Linear(hidden_dim * 2 * 2, self.num_classes)

    def _unpack_batch(self, batch):
        return batch

    def forward(self, x):

        x = x.long()
        x = torch.clamp(x, 0, self.vocab_size - 1)

        emb = self.embedding(x)

        out, hidden = self.rnn(emb)

        mean_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        x = torch.cat([mean_pool, max_pool], dim=1)

        x = self.dropout(x)

        return self.fc(x)

    def fit(self, train_loader, test_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-2
        )

        loss_fn = nn.CrossEntropyLoss()

        best_f1 = 0
        patience_counter = 0

        self.history = {"epoch": [], "loss": [], "acc": [], "f1": []}

        for epoch in range(self.max_epoch):

            self.train()
            total_loss = 0

            for batch_idx, (x, labels) in enumerate(train_loader):

                if batch_idx > 200:
                    break

                x = x.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                logits = self.forward(x)
                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, batch_idx)

            self.eval()

            y_true, y_pred = [], []

            with torch.no_grad():
                for x, labels in test_loader:

                    x = x.to(device)
                    labels = labels.to(device).long()

                    logits = self.forward(x)
                    preds = torch.argmax(logits, dim=1)

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc = np.mean(y_true == y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            self.history["epoch"].append(epoch)
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(acc)
            self.history["f1"].append(f1)

            print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")

            if f1 > best_f1 + 0.001:
                best_f1 = f1
                best_model = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_f1 > 0:
            self.load_state_dict(best_model)

    def run(self):

        train_loader = self.data["train_loader"]
        test_loader = self.data["test_loader"]

        self.fit(train_loader, test_loader)

        device = next(self.parameters()).device

        y_true, y_pred = [], []

        self.eval()

        with torch.no_grad():
            for x, labels in test_loader:

                x = x.to(device)
                labels = labels.to(device).long()

                logits = self.forward(x)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        return {
            "pred_y": np.array(y_pred),
            "true_y": np.array(y_true),
            "history": self.history
        }