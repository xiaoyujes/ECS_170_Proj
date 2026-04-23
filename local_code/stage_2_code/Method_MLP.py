# Arya Chaudhari Made this Model

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
<<<<<<<< HEAD:local_code/stage_2_code/Method_MLP_fullbatch_main.py
    # it defines the max rounds to train the model
    max_epoch = 250
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 5e-4
========
    max_epoch = 300   # good balance for your laptop
    learning_rate = 6e-4
>>>>>>>> a7220c7d1e9244dd4d1c6a773fb1c83473d2030f:local_code/stage_2_code/Method_MLP.py

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
<<<<<<<< HEAD:local_code/stage_2_code/Method_MLP_fullbatch_main.py
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 512)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 256)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(256, 128)
        self.activation_func_3 = nn.ReLU()
        self.fc_layer_4 = nn.Linear(128, 10)
        self.activation_func_4 = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.15)
========

        # --- Architecture (same as your working version) ---
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 10)

>>>>>>>> a7220c7d1e9244dd4d1c6a773fb1c83473d2030f:local_code/stage_2_code/Method_MLP.py

        self.historical_loss = []

    def forward(self, x):
<<<<<<<< HEAD:local_code/stage_2_code/Method_MLP_fullbatch_main.py
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        h = self.dropout(h)
        h = self.activation_func_2(self.fc_layer_2(h))
        h = self.dropout(h)
        h = self.activation_func_3(self.fc_layer_3(h))
        h = self.dropout(h)
        y_pred = self.activation_func_4(self.fc_layer_4(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
========
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)   # logits (correct for CrossEntropyLoss)
        return x
>>>>>>>> a7220c7d1e9244dd4d1c6a773fb1c83473d2030f:local_code/stage_2_code/Method_MLP.py

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

<<<<<<<< HEAD:local_code/stage_2_code/Method_MLP_fullbatch_main.py
        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
========
        # FULL BATCH (required by assignment)
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):

            preds = self.forward(X)
            loss = loss_fn(preds, y)
>>>>>>>> a7220c7d1e9244dd4d1c6a773fb1c83473d2030f:local_code/stage_2_code/Method_MLP.py

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


<<<<<<<< HEAD:local_code/stage_2_code/Method_MLP_fullbatch_main.py
            if train_loss.item() < best_loss:
                best_loss = train_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            if epoch%10 == 0: #originally epoch%100 but did epoch%1 just to get an idea of what is going on (change back when have stable model version)
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                train_accuracy = accuracy_evaluator.evaluate()
                #storing accuracy for roc curve / auc
                self.historical_accuracy.append(train_accuracy)
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
========
            self.historical_loss.append(loss.item())

            if epoch % 10 == 0:
                acc = (preds.argmax(1) == y).float().mean().item()
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | Accuracy {acc:.4f}")

>>>>>>>> a7220c7d1e9244dd4d1c6a773fb1c83473d2030f:local_code/stage_2_code/Method_MLP.py
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