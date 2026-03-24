"""
Neural Network Models for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, OrderedDict


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/FashionMNIST
    Architecture: Conv -> Conv -> FC -> FC
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by 2
        
        # Fully connected layers
        # Calcul dynamique de la taille après 2 poolings
        fc_size = (input_size // 4) ** 2 * 64  # // 4 car 2 poolings de stride 2
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self._fc_size = fc_size
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        
        # Flatten
        x = x.view(-1, self._fc_size)  # -> [batch, 3136]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for layer-wise strategies"""
        return [name for name, _ in self.named_parameters()]
    
    def get_layer_weights(self, layer_names: List[str] = None) -> OrderedDict:
        """
        Get weights of specific layers
        
        Args:
            layer_names: List of layer names to extract. If None, returns all.
            
        Returns:
            OrderedDict of layer weights
        """
        if layer_names is None:
            return self.state_dict()
        
        state_dict = self.state_dict()
        selected_weights = OrderedDict()
        
        for name in layer_names:
            if name in state_dict:
                selected_weights[name] = state_dict[name]
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model")
        
        return selected_weights
    
    def set_layer_weights(self, weights: OrderedDict):
        """
        Update specific layer weights
        
        Args:
            weights: OrderedDict of layer weights to update
        """
        current_state = self.state_dict()
        
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model, skipping")
        
        self.load_state_dict(current_state, strict=False)


class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST
    Classic CNN architecture
    """
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
        x = self.fc3(x)
        return x
    
    def get_layer_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]
    
    def get_layer_weights(self, layer_names: List[str] = None) -> OrderedDict:
        if layer_names is None:
            return self.state_dict()
        
        state_dict = self.state_dict()
        selected_weights = OrderedDict()
        
        for name in layer_names:
            if name in state_dict:
                selected_weights[name] = state_dict[name]
        
        return selected_weights
    
    def set_layer_weights(self, weights: OrderedDict):
        current_state = self.state_dict()
        
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
        
        self.load_state_dict(current_state, strict=False)


class CIFAR100CNN(nn.Module):
    """
    CNN for CIFAR-100 (100 classes, 32x32 RGB images)
    Deeper architecture suitable for more complex classification
    """
    def __init__(self, num_classes: int = 100):
        super(CIFAR100CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        # After 3 poolings: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 -> 4
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4 -> 2
        
        # Flatten
        x = x.view(-1, 512 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for layer-wise strategies"""
        return [name for name, _ in self.named_parameters()]
    
    def get_layer_weights(self, layer_names: List[str] = None) -> OrderedDict:
        """Get weights of specific layers"""
        if layer_names is None:
            return self.state_dict()
        
        state_dict = self.state_dict()
        selected_weights = OrderedDict()
        
        for name in layer_names:
            if name in state_dict:
                selected_weights[name] = state_dict[name]
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model")
        
        return selected_weights
    
    def set_layer_weights(self, weights: OrderedDict):
        """Update specific layer weights"""
        current_state = self.state_dict()
        
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model, skipping")
        
        self.load_state_dict(current_state, strict=False)


class EMNISTCNN(nn.Module):
    """
    CNN for EMNIST (47 classes, 28x28 grayscale images)
    Similar to SimpleCNN but adapted for more classes
    """
    def __init__(self, num_classes: int = 47):
        super(EMNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 poolings: 28 -> 14 -> 7
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch, 128, 3, 3]
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for layer-wise strategies"""
        return [name for name, _ in self.named_parameters()]
    
    def get_layer_weights(self, layer_names: List[str] = None) -> OrderedDict:
        """Get weights of specific layers"""
        if layer_names is None:
            return self.state_dict()
        
        state_dict = self.state_dict()
        selected_weights = OrderedDict()
        
        for name in layer_names:
            if name in state_dict:
                selected_weights[name] = state_dict[name]
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model")
        
        return selected_weights
    
    def set_layer_weights(self, weights: OrderedDict):
        """Update specific layer weights"""
        current_state = self.state_dict()
        
        for name, param in weights.items():
            if name in current_state:
                current_state[name] = param
            else:
                print(f"⚠️  Warning: Layer '{name}' not found in model, skipping")
        
        self.load_state_dict(current_state, strict=False)


DATASET_CONFIG = {
    'MNIST':        ('SimpleCNN',   10,  1, 28),
    'FashionMNIST': ('SimpleCNN',   10,  1, 28),
    'CIFAR10':      ('SimpleCNN',   10,  3, 32),
    'CIFAR100':     ('CIFAR100CNN', 100, 3, 32),
    'EMNIST':       ('EMNISTCNN',   47,  1, 28),
}

def create_model(model_name: str, num_classes: int = 10, in_channels: int = 1, input_size: int = 28) -> nn.Module:
    if model_name == 'SimpleCNN':
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    elif model_name == 'CIFAR100CNN':
        return CIFAR100CNN(num_classes=num_classes)
    elif model_name == 'EMNISTCNN':
        return EMNISTCNN(num_classes=num_classes)
    elif model_name == 'LeNet5':
        return LeNet5(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def create_model_for_dataset(dataset_name: str) -> nn.Module:
    """Crée le bon modèle pour un dataset donné"""
    config = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))
    model_name, num_classes, in_channels, input_size = config
    return create_model(model_name, num_classes=num_classes, in_channels=in_channels, input_size=input_size)


# Test function
if __name__ == "__main__":
    """Test model creation and layer extraction"""
    print("🧪 Testing SimpleCNN model...\n")
    
    # Create model
    model = create_model("SimpleCNN", num_classes=10)
    
    # Test forward pass
    print("\n📊 Testing forward pass...")
    dummy_input = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test layer extraction
    print("\n🔍 Layer names:")
    layer_names = model.get_layer_names()
    for i, name in enumerate(layer_names, 1):
        print(f"   {i}. {name}")
    
    # Test partial layer extraction
    print("\n🎯 Testing partial layer extraction...")
    first_two_layers = layer_names[:2]
    weights = model.get_layer_weights(first_two_layers)
    print(f"   Extracted {len(weights)} layers: {list(weights.keys())}")
    
    print("\n✅ All model tests passed!")