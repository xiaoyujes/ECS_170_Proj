
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset_Loader_CIFAR(Dataset):
    """Dataset class for the provided CIFAR-10 pickle file."""

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        limit: Optional[int] = None,
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self.normalize = normalize
        self.augment = augment and split == "train"

        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'.")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Could not find CIFAR file: {self.data_path}")

        with self.data_path.open("rb") as f:
            dataset = pickle.load(f)

        if split not in dataset:
            raise KeyError(f"The CIFAR file does not contain split '{split}'.")

        records = dataset[split]
        if limit is not None:
            records = records[:limit]

        images = []
        labels = []
        for instance in records:
            image = np.asarray(instance["image"], dtype=np.float32)  # H x W x C
            label = int(instance["label"])

            if image.shape != (32, 32, 3):
                raise ValueError(f"Expected image shape (32, 32, 3), got {image.shape}.")

            if self.normalize:
                image = image / 255.0

            # PyTorch Conv2d expects C x H x W, not H x W x C.
            image = np.transpose(image, (2, 0, 1))
            images.append(image)
            labels.append(label)

        self.images = torch.tensor(np.stack(images), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]

        if self.augment:
            # Simple CIFAR augmentation: random horizontal flip.
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, dims=[2])

        return image, label


def get_cifar_dataloaders(
    data_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for CIFAR-10."""

    train_dataset = Dataset_Loader_CIFAR(
        data_path=data_path,
        split="train",
        limit=train_limit,
        augment=augment,
    )
    test_dataset = Dataset_Loader_CIFAR(
        data_path=data_path,
        split="test",
        limit=test_limit,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage:
    # python Dataset_Loader_CIFAR.py
    default_path = Path(__file__).resolve().parent / "CIFAR"
    loader = Dataset_Loader_CIFAR(default_path, split="train", limit=5)
    print("Number of examples:", len(loader))
    print("Image tensor shape:", loader[0][0].shape)  # should be torch.Size([3, 32, 32])
    print("Label:", loader[0][1].item())
