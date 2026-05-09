from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from Dataset_Loader_CIFAR import get_cifar_dataloaders
from Method_CNN_CIFAR import Method_CNN_CIFAR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return running_loss / total_examples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": running_loss / total_examples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_learning_curve(history: Dict[str, List[float]], output_path: Path, title: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history["test_loss"], marker="o", label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(args, variant: str) -> Dict[str, float]:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_cifar_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        augment=args.augment,
    )

    model = Method_CNN_CIFAR(num_classes=10, variant=variant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    print(f"\nRunning variant: {variant}")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} | "
            f"test loss={test_metrics['loss']:.4f} | "
            f"test accuracy={test_metrics['accuracy']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, criterion, device)
    final_metrics.update(
        {
            "variant": variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_limit": args.train_limit if args.train_limit is not None else "full",
            "test_limit": args.test_limit if args.test_limit is not None else "full",
        }
    )

    # Save learning curve and metrics.
    curve_name = "cifar_loss_curve.png" if not args.compare and variant == args.variant else f"cifar_loss_curve_{variant}.png"
    plot_learning_curve(history, output_dir / curve_name, f"CIFAR-10 CNN Learning Curve ({variant})")

    with (output_dir / f"cifar_metrics_{variant}.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "final_metrics": final_metrics}, f, indent=2)

    torch.save(model.state_dict(), output_dir / f"cnn_cifar_{variant}.pt")
    return final_metrics


def save_result_table(results: List[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "CIFAR_result_table.csv"
    md_path = output_dir / "CIFAR_result_table.md"

    columns = [
        "variant",
        "epochs",
        "batch_size",
        "learning_rate",
        "train_limit",
        "test_limit",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in columns})

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# CIFAR-10 Result Table\n\n")
        f.write("| Variant | Epochs | Batch Size | Learning Rate | Train Set | Test Set | Accuracy | Precision | Recall | F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {row['variant']} | {row['epochs']} | {row['batch_size']} | {row['learning_rate']} | "
                f"{row['train_limit']} | {row['test_limit']} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN models on the provided CIFAR-10 dataset.")
    parser.add_argument("--data-path", type=str, default="./CIFAR", help="Path to the provided CIFAR pickle file.")
    parser.add_argument("--output-dir", type=str, default="./cifar_outputs", help="Folder to save outputs.")
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "dropout", "deep", "kernel5"])
    parser.add_argument("--compare", action="store_true", help="Run multiple CNN configurations for comparison.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=None, help="Use a subset for quick testing. Omit for full training set.")
    parser.add_argument("--test-limit", type=int, default=None, help="Use a subset for quick testing. Omit for full testing set.")
    parser.add_argument("--augment", action="store_true", help="Use simple random horizontal flip on training data.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    variants = ["baseline", "dropout", "deep", "kernel5"] if args.compare else [args.variant]
    results = []
    for variant in variants:
        results.append(run_experiment(args, variant))

    save_result_table(results, Path(args.output_dir))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
