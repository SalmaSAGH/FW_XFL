"""
Pre-download MNIST dataset to shared volume
This fixes the slow startup when all clients download simultaneously
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets
import torch


def predownload_mnist(data_dir="./data"):
    """
    Download MNIST once to shared directory
    All Docker containers will use this cached copy
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Pre-downloading MNIST Dataset")
    print("="*70)
    print(f"Target directory: {data_path.absolute()}")
    
    # Download training set
    print("\n📥 Downloading MNIST training set...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )
    print(f"✅ Training set: {len(train_dataset)} samples")
    
    # Download test set
    print("\n📥 Downloading MNIST test set...")
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    print(f"✅ Test set: {len(test_dataset)} samples")
    
    print("\n" + "="*70)
    print("✅ MNIST dataset downloaded successfully!")
    print("="*70)
    print(f"\nDataset location: {data_path.absolute()}")
    print("All Docker containers will now use this cached copy.")
    print("\nYou can now run: docker-compose up -d")


if __name__ == "__main__":
    predownload_mnist()