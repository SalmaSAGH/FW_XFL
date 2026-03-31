"""
Dataset loading and partitioning for Federated Learning
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, List
import numpy as np
from pathlib import Path


class DatasetPartitioner:
    """
    Partition dataset across multiple clients
    Supports both IID and non-IID distributions
    """

    def __init__(self, dataset: Dataset, num_clients: int,
                 distribution: str = "iid", seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.distribution = distribution
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.client_indices = self._partition_data()

    def _partition_data(self) -> List[List[int]]:
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        if self.distribution == "iid":
            return self._partition_iid(indices)
        elif self.distribution == "non_iid":
            return self._partition_non_iid(indices)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _partition_iid(self, indices: np.ndarray) -> List[List[int]]:
        samples_per_client = len(indices) // self.num_clients
        client_indices = []
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices.append(indices[start_idx:end_idx].tolist())
        print(f"IID partition created: {samples_per_client} samples per client")
        return client_indices

    def _partition_non_iid(self, indices: np.ndarray) -> List[List[int]]:
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            labels = np.array(self.dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")

        num_classes = len(np.unique(labels))
        classes_per_client = 2

        sorted_indices = [indices[labels[indices] == i] for i in range(num_classes)]

        client_indices = []
        for i in range(self.num_clients):
            client_classes = [(i * classes_per_client + j) % num_classes
                              for j in range(classes_per_client)]
            client_data = []
            for class_idx in client_classes:
                client_data.extend(sorted_indices[class_idx].tolist())
            np.random.shuffle(client_data)
            client_indices.append(client_data)

        print(f"Non-IID partition created: each client has {classes_per_client} classes")
        return client_indices

    def get_client_dataset(self, client_id: int) -> Subset:
        if client_id >= self.num_clients:
            raise ValueError(
                f"Client ID {client_id} exceeds num_clients {self.num_clients}"
            )
        return Subset(self.dataset, self.client_indices[client_id])


# ── Module-level functions ────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True
) -> Dataset:
    """Load a dataset from torchvision."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if dataset_name in ["MNIST", "FashionMNIST"]:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "EMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "CIFAR10":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
    elif dataset_name == "CIFAR100":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=2),  # Reduced for speed
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_map = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "EMNIST": datasets.EMNIST,
    }

    if dataset_name == "EMNIST":
        dataset = dataset_map[dataset_name](
            root=data_dir, train=train, download=True,
            transform=transform, split='byclass'
        )
        # DEBUG: Print first sample shape for EMNIST
        if dataset_name == "EMNIST" and len(dataset) > 0:
            sample_img, _ = dataset[0]
            print(f"DEBUG dataset.py EMNIST first sample shape: {sample_img.shape}")
    else:
        dataset = dataset_map[dataset_name](
            root=data_dir, train=train, download=True, transform=transform
        )

    split_name = "training" if train else "test"
    print(f"{dataset_name} {split_name} set loaded: {len(dataset)} samples")
    if hasattr(dataset, 'classes'):
        print(f"   Number of classes: {len(dataset.classes)}")

    return dataset


def create_single_client_loader(
    dataset_name: str,
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    distribution: str = "iid",
    data_dir: str = "./data",
    seed: int = 42
) -> DataLoader:
    """
    Charge uniquement le DataLoader d'UN seul client.
    Beaucoup plus rapide que create_dataloaders car on ne crée pas
    les 40 loaders inutiles.
    """
    train_dataset = load_dataset(dataset_name, data_dir, train=True)
    partitioner = DatasetPartitioner(train_dataset, num_clients, distribution, seed)
    client_dataset = partitioner.get_client_dataset(client_id)
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )


def create_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int = 256,
    distribution: str = "iid",
    data_dir: str = "./data",
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create dataloaders for ALL clients + test set.
    Utilisé uniquement par le serveur pour évaluation globale.
    Les clients utilisent create_single_client_loader à la place.
    """
    train_dataset = load_dataset(dataset_name, data_dir, train=True)
    test_dataset = load_dataset(dataset_name, data_dir, train=False)

    partitioner = DatasetPartitioner(train_dataset, num_clients, distribution, seed)

    client_loaders = []
    for cid in range(num_clients):
        client_dataset = partitioner.get_client_dataset(cid)
        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        client_loaders.append(client_loader)
        print(f"   Client {cid}: {len(client_dataset)} samples, {len(client_loader)} batches")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"✅ Test set: {len(test_dataset)} samples, {len(test_loader)} batches")

    return client_loaders, test_loader