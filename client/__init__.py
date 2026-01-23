"""
Client package for XFL-RPiLab
"""

from .model import create_model, SimpleCNN, LeNet5
from .dataset import load_dataset, create_dataloaders
from .trainer import LocalTrainer
from .metrics import MetricsCollector
from .client import FLClient

__all__ = [
    'create_model',
    'SimpleCNN',
    'LeNet5',
    'load_dataset',
    'create_dataloaders',
    'LocalTrainer',
    'MetricsCollector',
    'FLClient'
]