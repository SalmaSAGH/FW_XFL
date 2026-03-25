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
            xfl_param: Parameter for XFL (reserved for future use)
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
        # For now, only all_layers is supported (FedAvg standard)
        # XFL variants (cyclic, sparsification, quantization) are handled by the XFL class
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

        # Aggregate each layer - FIXED: force torch.float32 dtype
        aggregated_weights = OrderedDict()
        
        # Validate input dtypes
        for layer_name, updates in layer_updates.items():
            for i, (layer_weight, _) in enumerate(updates):
                if not layer_weight.dtype == torch.float32:
                    print(f"⚠️  Converting layer '{layer_name}' (client {i}) "
                          f"from {layer_weight.dtype} → float32")
                    updates[i] = (layer_weight.float(), updates[i][1])
        
        for layer_name, updates in layer_updates.items():
            total_samples = sum(num_samples for _, num_samples in updates)
            # FIXED: Force float32 dtype regardless of input
            first_weight_dtype = updates[0][0].dtype
            aggregated_layer = torch.zeros_like(
                updates[0][0], dtype=torch.float32
            )

            for layer_weight, num_samples in updates:
                weight_factor = num_samples / total_samples
                aggregated_layer += layer_weight.float() * weight_factor

            aggregated_weights[layer_name] = aggregated_layer.float()

        print(f"✅ XFL aggregation complete: {len(aggregated_weights)} layers, all float32")
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
