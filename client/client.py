"""
ULTRA-FAST Federated Learning Client
Optimized for speed with minimal overhead
"""

import requests
import pickle
import base64
from collections import OrderedDict
import time
from typing import Dict, Any, Tuple
import torch

from .trainer import LocalTrainer
from .metrics import MetricsCollector
from .model import create_model
from .dataset import create_dataloaders

# Import gc for garbage collection
import gc


class FLClient:
    """Federated Learning Client - Speed Optimized"""
    
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader,
        server_url: str = "http://localhost:5000",
        optimizer: str = "sgd",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        local_epochs: int = 1,
        timeout: int = 300,
        sparsification_threshold: float = 0.01,
        quantization_bits: int = 8,
        dataset_name: str = "MNIST",
        num_clients: int = 40,
        batch_size: int = 32,
        distribution: str = "iid",
        data_dir: str = "./data"
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.timeout = timeout
        # Retry settings for robust connection handling
        self.max_retries = 5
        self.base_delay = 1

        # XFL compression parameters
        self.sparsification_threshold = sparsification_threshold
        self.quantization_bits = quantization_bits

        # Dataset configuration - will be updated from server
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.data_dir = data_dir
        self.current_dataset_name = dataset_name  # Track current dataset
        self.current_model_name = "SimpleCNN"  # Track current model
        self.current_distribution = distribution  # Track current distribution

        # Initialize trainer
        self.trainer = LocalTrainer(
            model=model,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(client_id=client_id)

        print(f"✅ FLClient {client_id} initialized ({len(train_loader.dataset)} samples)")
    
    def _cleanup_memory(self):
        """Clean up memory after training to prevent OOM"""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear any temporary variables
        if hasattr(self, '_quantization_meta'):
            delattr(self, '_quantization_meta')
    
    def _weights_to_base64(self, weights: OrderedDict) -> str:
        """Serialize weights to base64 string"""
        weights_bytes = pickle.dumps(weights)
        return base64.b64encode(weights_bytes).decode('utf-8')
    
    def _base64_to_weights(self, base64_str: str) -> OrderedDict:
        """Deserialize weights from base64 string"""
        weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
        return pickle.loads(weights_bytes)
    
    def _get_model_for_dataset(self, dataset_name: str) -> Tuple[str, int]:
        """Get appropriate model and num_classes for a dataset"""
        dataset_model_map = {
            'MNIST': ('SimpleCNN', 10),
            'FashionMNIST': ('SimpleCNN', 10),
            'CIFAR10': ('SimpleCNN', 10),
            'CIFAR100': ('CIFAR100CNN', 100),
            'EMNIST': ('EMNISTCNN', 47),
        }
        return dataset_model_map.get(dataset_name, ('SimpleCNN', 10))
    
    def reload_data_if_needed(self, dataset_name: str, distribution: str = None, verbose: bool = False) -> bool:
        """Reload data if dataset or distribution changed from server config"""
        needs_reload = False
        
        # Check if dataset changed
        if dataset_name != self.current_dataset_name:
            needs_reload = True
            if verbose:
                print(f"Client {self.client_id}: Dataset changed from {self.current_dataset_name} to {dataset_name}")
        
        # Check if distribution changed
        if distribution is not None and distribution != self.current_distribution:
            needs_reload = True
            if verbose:
                print(f"Client {self.client_id}: Distribution changed from {self.current_distribution} to {distribution}")
        
        if not needs_reload:
            return False
        
        # Update dataset and distribution
        if dataset_name != self.current_dataset_name:
            model_name, num_classes = self._get_model_for_dataset(dataset_name)
            
            if verbose:
                print(f"Client {self.client_id}: Creating new model {model_name} with {num_classes} classes")
            
            self.model = create_model(model_name, num_classes=num_classes)
            
            self.trainer = LocalTrainer(
                model=self.model,
                optimizer_name="sgd",
                learning_rate=0.01,
                momentum=0.9,
                weight_decay=0.0001
            )
            
            self.current_dataset_name = dataset_name
            self.current_model_name = model_name
        
        # Update distribution if changed
        if distribution is not None:
            self.distribution = distribution
            self.current_distribution = distribution
        
        if verbose:
            print(f"Client {self.client_id}: Loading new dataset {dataset_name} with distribution {self.distribution}...")
        
        client_loaders, _ = create_dataloaders(
            dataset_name=self.current_dataset_name,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            distribution=self.distribution,
            data_dir=self.data_dir,
            seed=42
        )
        
        self.train_loader = client_loaders[self.client_id]
        
        if verbose:
            print(f"Client {self.client_id}: Data reloaded ({len(self.train_loader.dataset)} samples)")
        
        self._cleanup_memory()
        return True
    
    def participate_in_round(self, verbose: bool = False) -> bool:
        """
        Participate in one FL round - SPEED OPTIMIZED with XFL support

        Args:
            verbose: Print progress messages

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Get global model (with timeout)
            response = requests.get(
                f"{self.server_url}/api/get_global_model",
                timeout=10
            )
            if response.status_code != 200:
                return False

            data = response.json()
            weights = self._base64_to_weights(data['weights'])
            current_round = data['round']
            xfl_strategy = data.get('xfl_strategy', 'all_layers')
            xfl_param = data.get('xfl_param', 3)
            
            # NEW: Get dataset_name and data_distribution from server and reload data if needed
            server_dataset = data.get('dataset_name', 'MNIST')
            server_distribution = data.get('data_distribution', 'iid')
            self.reload_data_if_needed(server_dataset, server_distribution, verbose)

            if verbose:
                print(f"Client {self.client_id}: Round {current_round} - Model downloaded")
                print(f"   XFL Strategy: {xfl_strategy}, Dataset: {server_dataset}, Distribution: {server_distribution}")

            # Step 2: Update local model
            self.trainer.set_model_weights(weights)

            # Step 3: Train locally (THIS IS THE SLOW PART)
            start_train = time.time()
            training_metrics = self.trainer.train(
                train_loader=self.train_loader,
                num_epochs=self.local_epochs,
                verbose=False  # Disable verbose to save time
            )
            train_time = time.time() - start_train

            if verbose:
                print(f"Client {self.client_id}: Training done in {train_time:.1f}s "
                      f"(Loss: {training_metrics['loss']:.4f}, "
                      f"Acc: {training_metrics['accuracy']:.1f}%)")

            # Step 4: Get trained weights
            trained_weights = self.trainer.get_model_weights()
            num_samples = len(self.train_loader.dataset)

            # Step 5: Apply XFL strategy - select layers to send
            weights_to_send = self._apply_xfl_strategy(
                trained_weights, xfl_strategy, xfl_param, current_round, verbose
            )

            # Step 6: Collect minimal metrics (don't waste time)
            model_size_bytes = sum(p.numel() * p.element_size() for p in weights_to_send.values())

            # Simulate network metrics for dashboard display
            import random
            latency_ms = random.uniform(5, 50)
            packet_loss_rate = random.uniform(0, 0.05)
            jitter_ms = random.uniform(1, 10)

            full_metrics = self.metrics_collector.collect_full_metrics(
                training_metrics=training_metrics,
                model_weights=weights_to_send,  # Use sent weights for metrics
                network_metrics={
                    "bytes_sent": model_size_bytes,
                    "bytes_received": model_size_bytes,
                    "transmission_time": 0,
                    "latency_ms": latency_ms,
                    "packet_loss_rate": packet_loss_rate,
                    "jitter_ms": jitter_ms
                }
            )

            # Step 7: Send update to server
            start_upload = time.time()
            weights_b64 = self._weights_to_base64(weights_to_send)

            payload = {
                "client_id": self.client_id,
                "weights": weights_b64,
                "num_samples": num_samples,
                "metrics": full_metrics
            }

            # Include quantization metadata if quantization was applied
            if hasattr(self, '_quantization_meta') and self._quantization_meta:
                payload["quantization_meta"] = self._quantization_meta
                # Clear the metadata after sending
                delattr(self, '_quantization_meta')

            response = requests.post(
                f"{self.server_url}/api/submit_update",
                json=payload,
                timeout=30
            )

            upload_time = time.time() - start_upload

            if response.status_code != 200:
                return False

            result = response.json()

            if verbose:
                print(f"Client {self.client_id}: Upload done in {upload_time:.1f}s "
                      f"({result.get('submissions')}/{result.get('submissions')} received)")

            # Clean up memory after successful upload
            self._cleanup_memory()

            return True

        except Exception as e:
            if verbose:
                print(f"Client {self.client_id}: Error - {e}")
            # Clean up memory even on failure
            self._cleanup_memory()
            return False

    def _apply_sparsification(self, weights: OrderedDict, threshold: float) -> OrderedDict:
        """
        Apply sparsification to weights by setting small values to zero

        Args:
            weights: Weights to sparsify
            threshold: Threshold below which to set to zero

        Returns:
            Sparsified weights
        """
        sparsified = OrderedDict()
        for name, param in weights.items():
            sparsified[name] = torch.where(torch.abs(param) < threshold, torch.zeros_like(param), param)
        return sparsified

    def _apply_quantization(self, weights: OrderedDict, bits: int) -> Tuple[OrderedDict, Dict[str, Dict]]:
        """
        Apply quantization to reduce weight precision with metadata for dequantization

        Args:
            weights: Weights to quantize
            bits: Number of bits for quantization

        Returns:
            Tuple of (quantized weights, quantization metadata)
        """
        quantized = OrderedDict()
        quantization_meta = {}

        for name, param in weights.items():
            # Calculate quantization parameters
            min_val = param.min().item()
            max_val = param.max().item()

            if max_val == min_val:
                # No quantization needed for constant tensors
                quantized[name] = param
                quantization_meta[name] = {
                    'quantized': False,
                    'min_val': min_val,
                    'max_val': max_val,
                    'scale': 1.0,
                    'zero_point': 0
                }
                continue

            # Symmetric quantization around zero for better accuracy
            abs_max = max(abs(min_val), abs(max_val))
            scale = (2 * abs_max) / (2**bits - 1)
            zero_point = 0

            # Quantize: map to integers and keep as integers for bandwidth saving
            quantized_int = torch.round(param / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1).to(torch.int32)
            quantized[name] = quantized_int

            quantization_meta[name] = {
                'quantized': True,
                'min_val': min_val,
                'max_val': max_val,
                'scale': scale,
                'zero_point': zero_point,
                'bits': bits
            }

        return quantized, quantization_meta

    def _apply_xfl_strategy(
        self,
        trained_weights: OrderedDict,
        xfl_strategy: str,
        xfl_param: int,
        current_round: int,
        verbose: bool = False
    ) -> OrderedDict:
        """
        Apply XFL strategy to select which layers to send

        Args:
            trained_weights: Full trained model weights
            xfl_strategy: XFL strategy type
            xfl_param: XFL parameter
            current_round: Current round number
            verbose: Print selection info

        Returns:
            Selected weights to send
        """
        all_layer_names = list(trained_weights.keys())

        if xfl_strategy == "all_layers":
            return trained_weights

        elif xfl_strategy in ["first_n_layers", "last_n_layers", "random_layers"]:
            # For FedAvg variants, send full weights (server handles selection)
            return trained_weights

        elif xfl_strategy.startswith("xfl"):
            # For XFL, send all parameters of one layer per client (cyclic selection)
            # Group parameters by layer prefix (e.g., conv1, conv2, fc1, fc2)
            layer_prefixes = sorted(set(name.split('.')[0] for name in all_layer_names))
            num_layers = len(layer_prefixes)
            layer_index = (self.client_id + current_round - 1) % num_layers
            selected_prefix = layer_prefixes[layer_index]

            # Select all parameters that belong to this layer
            selected_weights = OrderedDict()
            for param_name in all_layer_names:
                if param_name.startswith(selected_prefix + '.'):
                    selected_weights[param_name] = trained_weights[param_name]

            if verbose:
                print(f"   📌 XFL: Sending layer {layer_index} ({selected_prefix}) - {len(selected_weights)} parameters")

            # Apply compression based on variant
            if xfl_strategy == "xfl_sparsification":
                selected_weights = self._apply_sparsification(selected_weights, self.sparsification_threshold)
                if verbose:
                    print(f"   🗜️ Applied sparsification with threshold {self.sparsification_threshold}")
            elif xfl_strategy == "xfl_quantization":
                selected_weights, quantization_meta = self._apply_quantization(selected_weights, self.quantization_bits)
                # Store quantization metadata for sending to server
                self._quantization_meta = quantization_meta
                if verbose:
                    print(f"   🗜️ Applied quantization to {self.quantization_bits} bits")

            return selected_weights

        else:
            # Default: send all layers
            if verbose:
                print(f"   ⚠️ Unknown XFL strategy: {xfl_strategy}, sending all layers")
            return trained_weights
