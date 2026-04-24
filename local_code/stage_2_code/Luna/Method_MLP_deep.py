import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
