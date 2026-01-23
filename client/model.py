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
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2 poolings: 28->14->7
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)  # -> [batch, 3136]
        
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


def create_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model (SimpleCNN, LeNet5)
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    models = {
        "SimpleCNN": SimpleCNN,
        "LeNet5": LeNet5
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name](num_classes=num_classes)
    print(f"✅ Model '{model_name}' created with {num_classes} classes")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


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