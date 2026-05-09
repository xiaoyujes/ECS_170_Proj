'''
Source code: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

from local_code.base_class.method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 25 * 20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 40)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Method_CNN_ORL(method):
    data = None
    max_epoch = 200
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.historical_loss = []

    def train(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = X_tensor.unsqueeze(1)
        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        trainloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        self.optimizer = optim.Adam(self.net.parameters(),
                                   lr=self.learning_rate)

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10}')
                    running_loss = 0.0
                    self.historical_loss.append(loss.item())

            if (epoch + 1) % 10 == 0:
                epoch_accuracy = correct / total
                print(f'*** Epoch [{epoch + 1}/{self.max_epoch}] Training Accuracy: {epoch_accuracy} ***')

        print('Finished Training')

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = X_tensor.unsqueeze(1)

        self.net.eval()
        predictions = []

        dataset = TensorDataset(X_tensor)
        testloader = DataLoader(dataset, batch_size=4, shuffle=False)

        with torch.no_grad():
            for data in testloader:
                images = data[0]
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.numpy())

        return np.array(predictions)

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        true_y = self.data['test']['y']
        correct = (pred_y == true_y).sum()
        total = len(true_y)
        accuracy = 100 * correct // total
        print(f'Accuracy of the network on the {total} test images: {accuracy}%')

        return {'pred_y': pred_y, 'true_y': true_y}