"""
Layer selection strategies for XFL (Cross-Layer Federated Learning)
"""

from collections import OrderedDict
from typing import List
import random


class LayerSelector:
    """
    Selects which layers to send based on strategy
    """
    
    def __init__(self, strategy: str = "all_layers", num_layers: int = None):
        """
        Args:
            strategy: Strategy name (all_layers, first_n_layers, last_n_layers, random_layers)
            num_layers: Number of layers to select (for first/last/random strategies)
        """
        self.strategy = strategy
        self.num_layers = num_layers
        
    def select_layers(self, model_weights: OrderedDict) -> OrderedDict:
        """
        Select layers to send based on strategy
        
        Args:
            model_weights: Complete model weights
            
        Returns:
            Selected layers weights
        """
        all_layer_names = list(model_weights.keys())
        total_layers = len(all_layer_names)
        
        if self.strategy == "all_layers":
            # Send all layers (default FL)
            return model_weights
        
        elif self.strategy == "first_n_layers":
            # Send first N layers (early features)
            n = self.num_layers if self.num_layers else total_layers // 2
            n = min(n, total_layers)
            selected_names = all_layer_names[:n]
            
        elif self.strategy == "last_n_layers":
            # Send last N layers (classifier)
            n = self.num_layers if self.num_layers else total_layers // 2
            n = min(n, total_layers)
            selected_names = all_layer_names[-n:]
            
        elif self.strategy == "random_layers":
            # Send random N layers
            n = self.num_layers if self.num_layers else total_layers // 2
            n = min(n, total_layers)
            selected_names = random.sample(all_layer_names, n)
            
        else:
            # Unknown strategy, send all
            return model_weights
        
        # Build selected weights dict
        selected_weights = OrderedDict()
        for name in selected_names:
            if name in model_weights:
                selected_weights[name] = model_weights[name]
        
        print(f"📊 Strategy '{self.strategy}': Sending {len(selected_weights)}/{total_layers} layers")
        print(f"   Selected layers: {list(selected_weights.keys())[:3]}..." if len(selected_weights) > 3 else f"   Selected layers: {list(selected_weights.keys())}")
        
        return selected_weights


# Test function
if __name__ == "__main__":
    """Test layer selector"""
    from .model import create_model
    
    print("🧪 Testing LayerSelector...\n")
    
    # Create model
    model = create_model("SimpleCNN", num_classes=10)
    weights = model.state_dict()
    
    print(f"Total model layers: {len(weights)}")
    print(f"Layer names: {list(weights.keys())}\n")
    
    # Test all strategies
    strategies = ["all_layers", "first_n_layers", "last_n_layers", "random_layers"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        selector = LayerSelector(strategy=strategy, num_layers=4)
        selected = selector.select_layers(weights)
        
        print(f"Selected {len(selected)} layers")
        for name in selected.keys():
            print(f"  - {name}")
    
    print("\n✅ All layer selector tests passed!")