from local_code.base_class.dataset import dataset
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import os


class CustomDataset(Dataset):

    def __init__(self, data, dataset_name):
        self.data = data
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]['image']
        label = self.data[idx]['label']

        if self.dataset_name == 'ORL':
            image = image[:, :, 0]
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            label = label - 1

        elif self.dataset_name == 'MNIST':
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        image = image / 255.0

        return image, label


class Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):

        print("loading data...")

        dataset_name = self.dataset_source_file_name[0]

        dataset_path = os.path.join(
            self.dataset_source_folder_path,
            dataset_name
        )

        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        train_dataset = CustomDataset(data["train"], dataset_name)
        test_dataset = CustomDataset(data["test"], dataset_name)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

        return {
            "train_loader": train_loader,
            "test_loader": test_loader
        }


def load_dataset(data_obj):

    loaded = data_obj.load()

    return loaded["train_loader"], loaded["test_loader"]