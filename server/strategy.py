"""
Aggregation strategies for Federated Learning with XFL support
"""

import torch
from collections import OrderedDict
from typing import List, Dict, Tuple
import random
import copy


class FedAvg:
    """
    Federated Averaging (FedAvg) aggregation strategy
    """
    
    def __init__(self, xfl_strategy: str = "all_layers", xfl_param: int = 3):
        """
        Args:
            xfl_strategy: XFL strategy type
                - "all_layers": Aggregate all layers (default FedAvg)
                - "first_n_layers": Aggregate first N layers only
                - "last_n_layers": Aggregate last N layers only
                - "random_layers": Aggregate random N layers
            xfl_param: Parameter for XFL (e.g., N for first_n/last_n/random)
        """
        self.name = "FedAvg"
        self.xfl_strategy = xfl_strategy
        self.xfl_param = xfl_param
        
        print(f"✅ {self.name} initialized with XFL strategy: {xfl_strategy}")
        if xfl_strategy != "all_layers":
            print(f"   XFL parameter: {xfl_param}")
    
    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_num_samples: List[int]
    ) -> OrderedDict:
        """
        Aggregate client model weights using FedAvg with XFL
        
        Args:
            client_weights: List of model weights from each client
            client_num_samples: List of number of samples per client
            
        Returns:
            Aggregated global model weights
        """
        if len(client_weights) == 0:
            raise ValueError("No client weights to aggregate")
        
        if len(client_weights) != len(client_num_samples):
            raise ValueError("Number of client weights must match number of sample counts")
        
        # Get all layer names
        all_layer_names = list(client_weights[0].keys())
        
        # Select layers based on XFL strategy
        selected_layers = self._select_layers(all_layer_names)
        
        print(f"📊 Aggregating {len(client_weights)} clients using {self.xfl_strategy}")
        print(f"   Total layers: {len(all_layer_names)}")
        print(f"   Aggregating: {len(selected_layers)} layers")
        
        # Calculate total samples
        total_samples = sum(client_num_samples)
        
        # Initialize aggregated weights
        aggregated_weights = OrderedDict()
        
        # Weighted aggregation for selected layers
        for name in selected_layers:
            aggregated_weights[name] = torch.zeros_like(client_weights[0][name])
            
            for client_weight, num_samples in zip(client_weights, client_num_samples):
                weight_factor = num_samples / total_samples
                if name in client_weight:
                    aggregated_weights[name] += client_weight[name] * weight_factor
        
        # For non-selected layers, use first client's weights (or could average)
        for name in all_layer_names:
            if name not in aggregated_weights:
                # Keep global model's version (from first client as reference)
                aggregated_weights[name] = client_weights[0][name].clone()
        
        return aggregated_weights
    
    def _select_layers(self, layer_names: List[str]) -> List[str]:
        """
        Select layers based on XFL strategy
        
        Args:
            layer_names: List of all layer names
            
        Returns:
            List of selected layer names to aggregate
        """
        if self.xfl_strategy == "all_layers":
            return layer_names
        
        elif self.xfl_strategy == "first_n_layers":
            # Select first N layers
            n = min(self.xfl_param * 2, len(layer_names))  # *2 for weight+bias
            selected = layer_names[:n]
            print(f"   📌 Selected first {len(selected)} layers: {selected[:3]}...")
            return selected
        
        elif self.xfl_strategy == "last_n_layers":
            # Select last N layers
            n = min(self.xfl_param * 2, len(layer_names))
            selected = layer_names[-n:]
            print(f"   📌 Selected last {len(selected)} layers: {selected[:3]}...")
            return selected
        
        elif self.xfl_strategy == "random_layers":
            # Select random N layers
            n = min(self.xfl_param * 2, len(layer_names))
            selected = random.sample(layer_names, n)
            print(f"   📌 Selected {len(selected)} random layers: {selected[:3]}...")
            return selected
        
        else:
            print(f"⚠️  Unknown XFL strategy: {self.xfl_strategy}, using all_layers")
            return layer_names
    
    def set_xfl_strategy(self, strategy: str, param: int = 3):
        """
        Change XFL strategy dynamically
        
        Args:
            strategy: New XFL strategy
            param: New XFL parameter
        """
        self.xfl_strategy = strategy
        self.xfl_param = param
        print(f"🔄 XFL strategy updated to: {strategy} (param={param})")
    
    def get_xfl_info(self) -> Dict:
        """
        Get current XFL configuration
        
        Returns:
            Dictionary with XFL info
        """
        return {
            "strategy": self.xfl_strategy,
            "param": self.xfl_param,
            "name": self.name
        }


class FedProx(FedAvg):
    """
    FedProx: Federated Optimization with Proximal Term
    Extension of FedAvg with regularization
    """

    def __init__(self, mu: float = 0.01, xfl_strategy: str = "all_layers", xfl_param: int = 3):
        super().__init__(xfl_strategy, xfl_param)
        self.name = "FedProx"
        self.mu = mu  # Proximal term coefficient
        print(f"⚠️  FedProx proximal term (μ={mu}) - client-side implementation needed")


class XFL(FedAvg):
    """
    eXtreme Federated Learning (XFL) strategy
    Clients send only one layer per round, selected cyclically
    """

    def __init__(self, xfl_variant: str = "cyclic", sparsification_threshold: float = 0.01, quantization_bits: int = 8):
        """
        Args:
            xfl_variant: XFL variant ("cyclic", "sparsification", "quantization")
            sparsification_threshold: Threshold for sparsification (if variant == "sparsification")
            quantization_bits: Number of bits for quantization (if variant == "quantization")
        """
        super().__init__(xfl_strategy="xfl", xfl_param=1)  # XFL sends 1 layer per client
        self.name = f"XFL-{xfl_variant.capitalize()}"
        self.xfl_variant = xfl_variant
        self.sparsification_threshold = sparsification_threshold
        self.quantization_bits = quantization_bits

        print(f"✅ {self.name} initialized")
        if xfl_variant == "sparsification":
            print(f"   Sparsification threshold: {sparsification_threshold}")
        elif xfl_variant == "quantization":
            print(f"   Quantization bits: {quantization_bits}")

    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_num_samples: List[int]
    ) -> OrderedDict:
        """
        Aggregate partial client weights (one layer per client) using XFL

        Args:
            client_weights: List of partial model weights from each client (one layer each)
            client_num_samples: List of number of samples per client

        Returns:
            Aggregated global model weights (only updated layers)
        """
        if len(client_weights) == 0:
            raise ValueError("No client weights to aggregate")

        if len(client_weights) != len(client_num_samples):
            raise ValueError("Number of client weights must match number of sample counts")

        # For XFL, each client sends exactly one layer
        # We need to aggregate per layer, weighted by samples
        layer_updates = {}  # layer_name -> list of (weight, num_samples)

        for client_weight, num_samples in zip(client_weights, client_num_samples):
            for layer_name, layer_weight in client_weight.items():
                if layer_name not in layer_updates:
                    layer_updates[layer_name] = []
                layer_updates[layer_name].append((layer_weight, num_samples))

        print(f"📊 Aggregating {len(client_weights)} clients with XFL-{self.xfl_variant}")
        print(f"   Layers received: {len(layer_updates)}")

        # Aggregate each layer
        aggregated_weights = OrderedDict()
        for layer_name, updates in layer_updates.items():
            total_samples = sum(num_samples for _, num_samples in updates)
            aggregated_layer = torch.zeros_like(updates[0][0])

            for layer_weight, num_samples in updates:
                weight_factor = num_samples / total_samples
                aggregated_layer += layer_weight * weight_factor

            aggregated_weights[layer_name] = aggregated_layer

        return aggregated_weights

    def get_xfl_info(self) -> Dict:
        """
        Get current XFL configuration

        Returns:
            Dictionary with XFL info
        """
        info = {
            "strategy": f"xfl_{self.xfl_variant}",
            "param": 1,  # Always 1 layer per client
            "name": self.name,
            "variant": self.xfl_variant,
            "sparsification_threshold": self.sparsification_threshold,
            "quantization_bits": self.quantization_bits
        }
        return info


def create_aggregation_strategy(
    strategy_name: str,
    xfl_strategy: str = "all_layers",
    xfl_param: int = 3
) -> FedAvg:
    """
    Factory function to create aggregation strategies with XFL

    Args:
        strategy_name: Name of the strategy ('fedavg', 'fedprox', 'xfl')
        xfl_strategy: XFL strategy type
        xfl_param: XFL parameter

    Returns:
        Aggregation strategy instance
    """
    strategies = {
        "fedavg": FedAvg,
        "fedprox": FedProx,
        "xfl": XFL
    }

    strategy_name_lower = strategy_name.lower()

    if strategy_name_lower not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    # For XFL, extract variant from xfl_strategy
    if strategy_name_lower == "xfl":
        # xfl_strategy should be like "xfl_cyclic", "xfl_sparsification", etc.
        if xfl_strategy.startswith("xfl_"):
            variant = xfl_strategy[4:]  # Remove "xfl_" prefix
            strategy = XFL(variant)
        else:
            # Default to cyclic if not specified
            strategy = XFL("cyclic")
    else:
        strategy = strategies[strategy_name_lower](
            xfl_strategy=xfl_strategy,
            xfl_param=xfl_param
        )

    print(f"✅ Aggregation strategy '{strategy.name}' created with XFL: {xfl_strategy}")

    return strategy


# Test function
if __name__ == "__main__":
    """Test XFL strategies"""
    print("🧪 Testing XFL Aggregation Strategies...\n")
    
    # Create dummy client weights
    print("📊 Creating dummy client weights (5 layers)...")
    
    client1_weights = OrderedDict([
        ("conv1.weight", torch.randn(32, 1, 3, 3)),
        ("conv1.bias", torch.randn(32)),
        ("conv2.weight", torch.randn(64, 32, 3, 3)),
        ("conv2.bias", torch.randn(64)),
        ("fc.weight", torch.randn(10, 64)),
        ("fc.bias", torch.randn(10)),
    ])
    
    client2_weights = OrderedDict([
        ("conv1.weight", torch.randn(32, 1, 3, 3)),
        ("conv1.bias", torch.randn(32)),
        ("conv2.weight", torch.randn(64, 32, 3, 3)),
        ("conv2.bias", torch.randn(64)),
        ("fc.weight", torch.randn(10, 64)),
        ("fc.bias", torch.randn(10)),
    ])
    
    client_weights = [client1_weights, client2_weights]
    client_num_samples = [100, 150]
    
    # Test each XFL strategy
    strategies_to_test = [
        ("all_layers", 3),
        ("first_n_layers", 1),  # First conv layer only
        ("last_n_layers", 1),   # FC layer only
        ("random_layers", 1),   # One random layer
    ]
    
    for xfl_strategy, xfl_param in strategies_to_test:
        print(f"\n{'='*70}")
        print(f"Testing XFL Strategy: {xfl_strategy} (param={xfl_param})")
        print(f"{'='*70}")
        
        fedavg = create_aggregation_strategy(
            "fedavg",
            xfl_strategy=xfl_strategy,
            xfl_param=xfl_param
        )
        
        aggregated = fedavg.aggregate(client_weights, client_num_samples)
        
        print(f"\n📊 Aggregated layers:")
        for name in aggregated.keys():
            print(f"   ✓ {name}: {tuple(aggregated[name].shape)}")
    
    print("\n✅ All XFL strategy tests completed!")