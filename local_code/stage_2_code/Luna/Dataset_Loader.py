import pandas as pd
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, header=None)

        self.y = data.iloc[:, 0].values
        self.X = data.iloc[:, 1:].values

        self.X = torch.tensor(self.X, dtype=torch.float32) / 255.0
        # 255.0으로 나눠서 정규화 하기 - 정규화 하고 나서 성능이 확 올라감.
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
