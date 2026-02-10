"""
ULTRA-FAST Federated Learning Client
Optimized for speed with minimal overhead
"""

import requests
import pickle
import base64
from collections import OrderedDict
import time
from typing import Dict, Any
import torch

from .trainer import LocalTrainer
from .metrics import MetricsCollector


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
        timeout: int = 300
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.timeout = timeout
        
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
    
    def _weights_to_base64(self, weights: OrderedDict) -> str:
        """Serialize weights to base64 string"""
        weights_bytes = pickle.dumps(weights)
        return base64.b64encode(weights_bytes).decode('utf-8')
    
    def _base64_to_weights(self, base64_str: str) -> OrderedDict:
        """Deserialize weights from base64 string"""
        weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
        return pickle.loads(weights_bytes)
    
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
                f"{self.server_url}/get_global_model",
                timeout=10
            )
            if response.status_code != 200:
                return False

            data = response.json()
            weights = self._base64_to_weights(data['weights'])
            current_round = data['round']
            xfl_strategy = data.get('xfl_strategy', 'all_layers')
            xfl_param = data.get('xfl_param', 3)

            if verbose:
                print(f"Client {self.client_id}: Round {current_round} - Model downloaded")
                print(f"   XFL Strategy: {xfl_strategy}")

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

            full_metrics = self.metrics_collector.collect_full_metrics(
                training_metrics=training_metrics,
                model_weights=weights_to_send,  # Use sent weights for metrics
                network_metrics={
                    "bytes_sent": model_size_bytes,
                    "bytes_received": model_size_bytes,
                    "transmission_time": 0
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

            response = requests.post(
                f"{self.server_url}/submit_update",
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

            return True

        except Exception as e:
            if verbose:
                print(f"Client {self.client_id}: Error - {e}")
            return False

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

            return selected_weights

        else:
            # Default: send all layers
            if verbose:
                print(f"   ⚠️ Unknown XFL strategy: {xfl_strategy}, sending all layers")
            return trained_weights
