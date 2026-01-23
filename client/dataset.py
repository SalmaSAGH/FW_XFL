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
    
    def __init__(self, dataset: Dataset, num_clients: int, distribution: str = "iid", seed: int = 42):
        """
        Args:
            dataset: PyTorch dataset to partition
            num_clients: Number of clients
            distribution: 'iid' or 'non_iid'
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.distribution = distribution
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.client_indices = self._partition_data()
        
    def _partition_data(self) -> List[List[int]]:
        """Partition data indices among clients"""
        
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
        """
        IID partition: randomly distribute samples equally among clients
        """
        samples_per_client = len(indices) // self.num_clients
        client_indices = []
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices.append(indices[start_idx:end_idx].tolist())
        
        print(f"✅ IID partition created: {samples_per_client} samples per client")
        return client_indices
    
    def _partition_non_iid(self, indices: np.ndarray) -> List[List[int]]:
        """
        Non-IID partition: each client gets samples from only 2 classes
        Simulates heterogeneous data distribution
        """
        # Get labels for all samples
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            labels = np.array(self.dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")
        
        num_classes = len(np.unique(labels))
        classes_per_client = 2  # Each client gets 2 classes
        
        # Sort indices by label
        sorted_indices = [indices[labels[indices] == i] for i in range(num_classes)]
        
        client_indices = []
        
        for i in range(self.num_clients):
            # Assign 2 classes to this client
            client_classes = [(i * classes_per_client + j) % num_classes 
                            for j in range(classes_per_client)]
            
            client_data = []
            for class_idx in client_classes:
                client_data.extend(sorted_indices[class_idx].tolist())
            
            np.random.shuffle(client_data)
            client_indices.append(client_data)
        
        print(f"✅ Non-IID partition created: each client has {classes_per_client} classes")
        return client_indices
    
    def get_client_dataset(self, client_id: int) -> Subset:
        """
        Get dataset subset for a specific client
        
        Args:
            client_id: ID of the client (0 to num_clients-1)
            
        Returns:
            Subset of the dataset for this client
        """
        if client_id >= self.num_clients:
            raise ValueError(f"Client ID {client_id} exceeds num_clients {self.num_clients}")
        
        return Subset(self.dataset, self.client_indices[client_id])


def load_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True
) -> Dataset:
    """
    Load a dataset from torchvision
    
    Args:
        dataset_name: Name of dataset (MNIST, CIFAR10, FashionMNIST)
        data_dir: Directory to store/load data
        train: If True, load training set; else test set
        
    Returns:
        PyTorch Dataset
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Define transforms
    if dataset_name in ["MNIST", "FashionMNIST"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load dataset
    dataset_map = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10
    }
    
    dataset_class = dataset_map[dataset_name]
    dataset = dataset_class(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    split_name = "training" if train else "test"
    print(f"✅ {dataset_name} {split_name} set loaded: {len(dataset)} samples")
    
    return dataset


def create_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int = 32,
    distribution: str = "iid",
    data_dir: str = "./data",
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create dataloaders for federated learning
    
    Args:
        dataset_name: Name of dataset
        num_clients: Number of clients
        batch_size: Batch size for training
        distribution: 'iid' or 'non_iid'
        data_dir: Directory to store/load data
        seed: Random seed
        
    Returns:
        Tuple of (list of client dataloaders, test dataloader)
    """
    # Load training and test datasets
    train_dataset = load_dataset(dataset_name, data_dir, train=True)
    test_dataset = load_dataset(dataset_name, data_dir, train=False)
    
    # Partition training data among clients
    partitioner = DatasetPartitioner(
        train_dataset, 
        num_clients, 
        distribution, 
        seed
    )
    
    # Create dataloaders for each client
    client_loaders = []
    for client_id in range(num_clients):
        client_dataset = partitioner.get_client_dataset(client_id)
        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=False
        )
        client_loaders.append(client_loader)
        print(f"   Client {client_id}: {len(client_dataset)} samples, "
              f"{len(client_loader)} batches")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Test set: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return client_loaders, test_loader


# Test function
if __name__ == "__main__":
    """Test dataset loading and partitioning"""
    print("🧪 Testing dataset module...\n")
    
    # Test dataset loading
    print("📊 Loading MNIST dataset...")
    train_dataset = load_dataset("MNIST", train=True)
    test_dataset = load_dataset("MNIST", train=False)
    
    # Test IID partitioning
    print("\n📊 Testing IID partitioning...")
    client_loaders, test_loader = create_dataloaders(
        dataset_name="MNIST",
        num_clients=5,
        batch_size=32,
        distribution="iid"
    )
    
    # Test non-IID partitioning
    print("\n📊 Testing non-IID partitioning...")
    client_loaders_non_iid, _ = create_dataloaders(
        dataset_name="MNIST",
        num_clients=5,
        batch_size=32,
        distribution="non_iid"
    )
    
    # Test batch iteration
    print("\n📊 Testing batch iteration for Client 0...")
    client_0_loader = client_loaders[0]
    for batch_idx, (data, target) in enumerate(client_0_loader):
        if batch_idx == 0:
            print(f"   Batch shape: {data.shape}")
            print(f"   Labels shape: {target.shape}")
            print(f"   Labels in batch: {target.unique().tolist()}")
        break
    
    print("\n✅ All dataset tests passed!")