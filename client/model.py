"""
Neural Network Models for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, OrderedDict


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

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
        mismatches = []
        for name, param in weights.items():
            if name in current_state:
                if param.shape != current_state[name].shape:
                    mismatches.append(f"{name}: expected {current_state[name].shape} got {param.shape}")
                else:
                    current_state[name] = param
            else:
                mismatches.append(f"{name} ({param.shape}): missing in model")
        
        if mismatches:
            print(f"⚠️  Model set_layer_weights: {len(mismatches)} mismatches:")
            for m in mismatches[:3]:  # First 3 only
                print(f"   {m}")
            if len(mismatches) > 3:
                print(f"   ... +{len(mismatches)-3} more")
        
        self.load_state_dict(current_state, strict=False)



class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Compute size after two 5×5 convs and two 2×2 poolings
        conv_output_size = ((input_size - 4) // 2 - 4) // 2
        self._fc_size = 16 * conv_output_size * conv_output_size

        self.fc1 = nn.Linear(self._fc_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._fc_size)
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
        mismatches = []
        for name, param in weights.items():
            if name in current_state:
                if param.shape != current_state[name].shape:
                    mismatches.append(f"{name}: expected {current_state[name].shape} got {param.shape}")
                else:
                    current_state[name] = param
            else:
                mismatches.append(f"{name} ({param.shape}): missing")
        
        if mismatches:
            print(f"⚠️  Model set_layer_weights: {len(mismatches)} mismatches:")
            for m in mismatches[:3]:
                print(f"   {m}")
            if len(mismatches) > 3:
                print(f"   ... +{len(mismatches)-3} more")
        
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


class TinyCNN(nn.Module):
    """Tiny CNN for MNIST-like datasets on Raspberry Pi."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        fc_size = (input_size // 4) ** 2 * 16
        self.fc1 = nn.Linear(fc_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, (x.size(2) * x.size(3)) * x.size(1))
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
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


class MicroLeNet(nn.Module):
    """Compact LeNet-style network for MNIST/FashionMNIST."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(MicroLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        fc_size = (input_size // 4) ** 2 * 32
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, (x.size(2) * x.size(3)) * x.size(1))
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
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


class DepthwiseCNN(nn.Module):
    """Lightweight depthwise-separable CNN for CIFAR-10 and CIFAR-100."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, input_size: int = 32):
        super(DepthwiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dwconv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.pwconv2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.25)

        fc_size = (input_size // 4) ** 2 * 64
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.pwconv2(self.dwconv2(x)))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, (x.size(2) * x.size(3)) * x.size(1))
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
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
        self.load_state_dict(current_state, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW MODELS FOR FEDERATED LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class MobileNetV2(nn.Module):
    """MobileNetV2 - Optimized for edge devices like Raspberry Pi"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(MobileNetV2, self).__init__()
        
        # Adjust first conv for different input channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Inverted residual blocks
        self.layer1 = self._make_layer(32, 16, 1, stride=1)
        self.layer2 = self._make_layer(16, 24, 2, stride=2)   # 28→14
        self.layer3 = self._make_layer(24, 32, 3, stride=2)   # 14→7
        self.layer4 = self._make_layer(32, 64, 2, stride=2)   # 7→4
        self.layer5 = self._make_layer(64, 96, 3, stride=1)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))  # 28→14 or 32→16
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
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


class InvertedResidual(nn.Module):
    """Inverted Residual block for MobileNetV2"""
    
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * 6  # Expansion factor
        
        layers = []
        # Expansion
        layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.stride == 1 and x.shape == out.shape:
            return x + out
        else:
            return out


class ResNet8(nn.Module):
    """ResNet-8 - Lightweight ResNet for edge devices"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(ResNet8, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


class ResidualBlock(nn.Module):
    """Basic Residual block for ResNet"""
    
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 - Very efficient for edge devices"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super(ShuffleNetV2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 24)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(24, 48, 2)
        self.stage3 = self._make_stage(48, 96, 2)
        self.stage4 = self._make_stage(96, 192, 2)
        
        self.conv5 = nn.Conv2d(192, 1024, kernel_size=1, bias=False)
        self.gn5 = nn.GroupNorm(4, 1024)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        
        self._init_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ShuffleBlock(in_channels, out_channels, 2))
        for _ in range(1, num_blocks):
            layers.append(ShuffleBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.gn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


class ShuffleBlock(nn.Module):
    """ShuffleNet block"""
    
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        
        mid_channels = out_channels // 2
        
        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        if self.stride == 2:
            # Downsample
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        else:
            # No downsample
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self.branch1(x1), x2], dim=1)
        
        # Channel shuffle
        out = self._channel_shuffle(out)
        return out
    
    def _channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // 2
        x = x.view(batch_size, channels_per_group, 2, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


# ── Dataset → model mapping ───────────────────────────────────────────────────
# (model_name, num_classes, in_channels, input_size)
DATASET_CONFIG = {
    'MNIST':        ('TinyCNN',     10,  1, 28),
    'FashionMNIST': ('MicroLeNet',  10,  1, 28),
    'CIFAR10':      ('DepthwiseCNN',10,  3, 32),
    'CIFAR100':     ('CIFAR100CNN', 100, 3, 32),
    'EMNIST':       ('EMNISTCNN',   47,  1, 28),
}


def get_model_params_for_dataset(model_name: str, dataset_name: str):
    """Retourne les paramètres corrects pour un modèle selon le dataset"""
    # Certains modèles ont des paramètres fixes
    fixed_params = {
        'CIFAR100CNN': (100, 3, 32),  # Toujours CIFAR-100
        'EMNISTCNN': (47, 1, 28),     # Toujours EMNIST
    }
    
    if model_name in fixed_params:
        num_classes, in_channels, input_size = fixed_params[model_name]
        return num_classes, in_channels, input_size
    
    # Pour les autres modèles, utiliser les paramètres du dataset
    _, num_classes, in_channels, input_size = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))
    return num_classes, in_channels, input_size


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
        model = LeNet5(num_classes=num_classes,
                       in_channels=in_channels,
                       input_size=input_size)
    elif model_name == 'TinyCNN':
        model = TinyCNN(num_classes=num_classes,
                        in_channels=in_channels,
                        input_size=input_size)
    elif model_name == 'MicroLeNet':
        model = MicroLeNet(num_classes=num_classes,
                           in_channels=in_channels,
                           input_size=input_size)
    elif model_name == 'DepthwiseCNN':
        model = DepthwiseCNN(num_classes=num_classes,
                             in_channels=in_channels,
                             input_size=input_size)
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(num_classes=num_classes,
                           in_channels=in_channels,
                           input_size=input_size)
    elif model_name == 'ResNet8':
        model = ResNet8(num_classes=num_classes,
                       in_channels=in_channels,
                       input_size=input_size)
    elif model_name == 'ShuffleNetV2':
        model = ShuffleNetV2(num_classes=num_classes,
                            in_channels=in_channels,
                            input_size=input_size)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: SimpleCNN, CIFAR100CNN, EMNISTCNN, LeNet5, TinyCNN, MicroLeNet, DepthwiseCNN, MobileNetV2, ResNet8, ShuffleNetV2")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model '{model_name}' created | {num_classes} classes | {total_params:,} params")
    return model


def create_model_for_dataset(dataset_name: str) -> nn.Module:
    """Crée le bon modèle pour un dataset donné via DATASET_CONFIG."""
    cfg = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))
    model_name, num_classes, in_channels, input_size = cfg
    return create_model(model_name, num_classes=num_classes,
                        in_channels=in_channels, input_size=input_size)