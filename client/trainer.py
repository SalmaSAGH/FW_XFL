"""
Local training module - SPEED OPTIMIZED
The training loop is the bottleneck - this fixes it
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
import time


class LocalTrainer:
    """
    Local model trainer - OPTIMIZED FOR SPEED
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str = "sgd",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create optimizer
        if optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ LocalTrainer initialized on device: {self.device}")
    
    def train(
        self,
        train_loader,
        num_epochs: int = 1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train model locally - OPTIMIZED
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                num_batches += 1
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        if verbose:
            print(f"✅ Training completed in {training_time:.2f}s")
            print(f"   Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "training_time": training_time,
            "num_batches": num_batches,
            "total_samples": total
        }
    
    def evaluate(
        self,
        test_loader,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        if verbose:
            print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def get_model_weights(self):
        """Get model weights"""
        return self.model.state_dict()
    
    def set_model_weights(self, weights):
        """Set model weights"""
        self.model.load_state_dict(weights)
    
    def get_num_samples(self, data_loader):
        """Get number of samples in data loader"""
        return len(data_loader.dataset)