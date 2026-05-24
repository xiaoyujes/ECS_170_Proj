from local_code.base_class.dataset import dataset
from torch.utils.data import Dataset, DataLoader
import torch
import os
import re
from collections import Counter


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"!{2,}", " ! ", text)
    text = re.sub(r"\?{2,}", " ? ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        tokens = tokenize(clean_text(text))
        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.items():

        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


class CustomDataset(Dataset):

    def __init__(self, data, vocab, max_len=128):

        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):

        tokens = tokenize(clean_text(text))

        ids = [
            self.vocab.get(tok, 1)
            for tok in tokens
        ]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        if len(ids) < self.max_len:
            ids += [0] * (
                self.max_len - len(ids)
            )

        return torch.tensor(
            ids,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        x = self.encode(item["text"])

        y = torch.tensor(
            item["label"],
            dtype=torch.long
        )

        return x, y


class Dataset_Loader(dataset):

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, verbose=False):

        base_path = os.path.join(
            self.dataset_source_folder_path,
            "text_classification"
        )

        train_texts = []
        test_texts = []

        train_data = []
        test_data = []

        for label_name, label in [("pos", 1), ("neg", 0)]:

            folder = os.path.join(
                base_path,
                "train",
                label_name
            )

            for file in os.listdir(folder):

                with open(
                    os.path.join(folder, file),
                    "r",
                    encoding="utf-8"
                ) as f:

                    text = f.read()

                    train_texts.append(text)

                    train_data.append({
                        "text": text,
                        "label": label
                    })

        for label_name, label in [("pos", 1), ("neg", 0)]:

            folder = os.path.join(
                base_path,
                "test",
                label_name
            )

            for file in os.listdir(folder):

                with open(
                    os.path.join(folder, file),
                    "r",
                    encoding="utf-8"
                ) as f:

                    text = f.read()

                    test_texts.append(text)

                    test_data.append({
                        "text": text,
                        "label": label
                    })

        vocab = build_vocab(train_texts)

        train_dataset = CustomDataset(
            train_data,
            vocab,
            max_len=120
        )

        test_dataset = CustomDataset(
            test_data,
            vocab,
            max_len=120
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )

        if verbose:
            print("Train size:", len(train_data))
            print("Test size:", len(test_data))
            print("Vocab size:", len(vocab))

        return train_loader, test_loader, vocab


def load_dataset(data_obj):

    train_loader, test_loader, vocab = data_obj.load()

    return (
        train_loader,
        test_loader,
        vocab
    )