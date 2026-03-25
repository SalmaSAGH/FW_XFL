"""
Neural Network Models for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, OrderedDict


class SimpleCNN(nn.Module):
    """CNN for MNIST/FashionMNIST/CIFAR10 — supports variable in_channels and input_size."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 poolings: input_size → input_size//2 → input_size//4
        fc_size = (input_size // 4) ** 2 * 64
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self._fc_size = fc_size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._fc_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_layer_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_layer_weights(self, layer_names=None):
        if layer_names is None:
            return self.state_dict()
        state_dict = self.state_dict()
        return OrderedDict((n, state_dict[n]) for n in layer_names if n in state_dict)

    def set_layer_weights(self, weights):
        current_state = self.state_dict()
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_layer_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_layer_weights(self, layer_names=None):
        if layer_names is None:
            return self.state_dict()
        state_dict = self.state_dict()
        return OrderedDict((n, state_dict[n]) for n in layer_names if n in state_dict)

    def set_layer_weights(self, weights):
        current_state = self.state_dict()
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


class CIFAR100CNN(nn.Module):
    """
    CNN for CIFAR-100 (100 classes, 32×32 RGB images).

    Architecture corrigée :
      Input 32×32
      → Conv1+BN1+Pool  → 16×16
      → Conv2+BN2+Pool  → 8×8
      → Conv3+BN3+Pool  → 4×4
      → Conv4+BN4       → 4×4  (PAS de pooling ici — évite le bug batch_size)
      → Flatten 512*4*4 = 8192
      → FC1(8192→1024) → FC2(1024→512) → FC3(512→num_classes)

    Avant, 4 poolings ramenaient à 2×2, mais fc1 attendait 4×4 → mismatch
    qui faisait interpréter un batch de 256 comme 1024 samples.
    """

    def __init__(self, num_classes: int = 100):
        super(CIFAR100CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Après 3 poolings : 32 → 16 → 8 → 4  (conv4 sans pool → reste 4×4)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32 → 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16 → 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 8  → 4
        x = F.relu(self.bn4(self.conv4(x)))               # 4  → 4 (pas de pool)

        x = x.view(-1, 512 * 4 * 4)                      # 8192 ✓

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_layer_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_layer_weights(self, layer_names=None):
        if layer_names is None:
            return self.state_dict()
        state_dict = self.state_dict()
        return OrderedDict((n, state_dict[n]) for n in layer_names if n in state_dict)

    def set_layer_weights(self, weights):
        current_state = self.state_dict()
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


class EMNISTCNN(nn.Module):
    """CNN for EMNIST (47 classes, 28×28 grayscale)."""

    def __init__(self, num_classes: int = 47):
        super(EMNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 28 → 14 → 7  (2 poolings; conv3 sans pool)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 → 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 → 7
        x = F.relu(self.conv3(x))               # 7  → 7 (pas de pool)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_layer_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_layer_weights(self, layer_names=None):
        if layer_names is None:
            return self.state_dict()
        state_dict = self.state_dict()
        return OrderedDict((n, state_dict[n]) for n in layer_names if n in state_dict)

    def set_layer_weights(self, weights):
        current_state = self.state_dict()
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


# ── Dataset → model mapping ───────────────────────────────────────────────────
# (model_name, num_classes, in_channels, input_size)
DATASET_CONFIG = {
    'MNIST':        ('SimpleCNN',   10,  1, 28),
    'FashionMNIST': ('SimpleCNN',   10,  1, 28),
    'CIFAR10':      ('SimpleCNN',   10,  3, 32),
    'CIFAR100':     ('CIFAR100CNN', 100, 3, 32),
    'EMNIST':       ('EMNISTCNN',   47,  1, 28),
}


def create_model(model_name: str, num_classes: int = 10,
                 in_channels: int = 1, input_size: int = 28) -> nn.Module:
    """Factory — crée le bon modèle avec les bons paramètres."""
    if model_name == 'SimpleCNN':
        model = SimpleCNN(num_classes=num_classes,
                          in_channels=in_channels,
                          input_size=input_size)
    elif model_name == 'CIFAR100CNN':
        model = CIFAR100CNN(num_classes=num_classes)
    elif model_name == 'EMNISTCNN':
        model = EMNISTCNN(num_classes=num_classes)
    elif model_name == 'LeNet5':
        model = LeNet5(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: SimpleCNN, CIFAR100CNN, EMNISTCNN, LeNet5")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model '{model_name}' created | {num_classes} classes | "
          f"{total_params:,} params")
    return model


def create_model_for_dataset(dataset_name: str) -> nn.Module:
    """Crée le bon modèle pour un dataset donné via DATASET_CONFIG."""
    cfg = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))
    model_name, num_classes, in_channels, input_size = cfg
    return create_model(model_name, num_classes=num_classes,
                        in_channels=in_channels, input_size=input_size)