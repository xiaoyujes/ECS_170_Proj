'''
Source code: https://pytorch.org/hub/pytorch_vision_resnet/
'''

from local_code.base_class.method import method
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class ResNetORL(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(ResNetORL, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class Method_ResNet_ORL(method):
    data = None
    max_epoch = 5
    learning_rate = 0.001

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        self.net = ResNetORL(num_classes=40, pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.historical_loss = []
        self.historical_accuracy = []

        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def preprocess_images(self, X):
        if len(X.shape) == 3:
            X = X.unsqueeze(1)

        batch_size = X.shape[0]

        X_rgb = torch.zeros((batch_size, 3, 224, 224))

        for i in range(batch_size):
            img = transforms.ToPILImage()(X[i])
            img = self.resize_transform(img)
            img_tensor = transforms.ToTensor()(img)
            img_rgb = img_tensor.repeat(3, 1, 1)
            X_rgb[i] = img_rgb

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X_rgb = (X_rgb - mean) / std

        return X_rgb

    def train(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = self.preprocess_images(X_tensor)
        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        trainloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

        self.optimizer = optim.Adam(self.net.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=1e-4)

        print(f'Training ResNet on ORL...')
        print(f'Input shape: {X_tensor.shape}')

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0
            epoch_loss_sum = 0.0
            num_batches = 0

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

                epoch_loss_sum += loss.item()
                num_batches += 1
                running_loss += loss.item()

                if i % 10 == 9:
                    batch_acc = 100 * correct / total
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}, acc: {batch_acc:.2f}%')
                    running_loss = 0.0

            avg_epoch_loss = epoch_loss_sum / num_batches
            self.historical_loss.append(avg_epoch_loss)

            epoch_acc = 100 * correct / total
            self.historical_accuracy.append(epoch_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f'*** Epoch [{epoch + 1}/{self.max_epoch}] Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% ***')

        print('Finished Training')

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = self.preprocess_images(X_tensor)

        self.net.eval()
        predictions = []

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor)
        testloader = DataLoader(dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for data in testloader:
                images = data[0]
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.numpy())

        return np.array(predictions)

    def run(self):
        print('method running...')
        print('--start training ResNet...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        true_y = self.data['test']['y']
        correct = (pred_y == true_y).sum()
        total = len(true_y)
        accuracy = 100 * correct // total
        print(f'ResNet Accuracy on {total} test images: {accuracy} %')

        return {'pred_y': pred_y, 'true_y': true_y}
