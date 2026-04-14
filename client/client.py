"""
ULTRA-FAST Federated Learning Client
Optimized for speed with minimal overhead
"""

import os
import requests
import pickle
import base64
from collections import OrderedDict
import time
from typing import Dict, Any, Tuple
import torch
import torch.optim as optim

from .trainer import LocalTrainer
from .metrics import MetricsCollector
from .model import create_model, DATASET_CONFIG
from .dataset import create_single_client_loader
from config.config_parser import load_config

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
        self.max_retries = 5
        self.base_delay = 1

        self.sparsification_threshold = sparsification_threshold
        self.quantization_bits = quantization_bits

        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.data_dir = data_dir
        self.current_dataset_name = dataset_name
        self.current_model_name = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))[0]
        self.current_distribution = distribution

        self.trainer = LocalTrainer(
            model=model,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config',
                'config.yaml'
            )
            if not os.path.exists(config_path):
                config_path = os.path.join(os.getcwd(), 'config', 'config.yaml')

            config = load_config(config_path)
            if hasattr(config.network, 'model_dump'):
                network_config = config.network.model_dump()
            else:
                network_config = config.network.dict()
        except Exception as e:
            print(f"⚠️ Could not load network config: {e}")
            network_config = {
                'simulate_constraints': True,
                'latency_ms': 50,
                'latency_std_ms': 10,
                'packet_loss_rate': 0.02,
                'jitter_ms': 5
            }

        self.metrics_collector = MetricsCollector(
            client_id=client_id,
            network_config=network_config
        )
        self.metrics_collector.start_collection()

        print(f"✅ FLClient {client_id} initialized | dataset={dataset_name} | "
              f"model={self.current_model_name} | {len(train_loader.dataset)} samples")

    def _cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if hasattr(self, '_quantization_meta'):
            delattr(self, '_quantization_meta')

    def _weights_to_base64(self, weights: OrderedDict) -> str:
        return base64.b64encode(pickle.dumps(weights)).decode('utf-8')

    def _base64_to_weights(self, base64_str: str) -> OrderedDict:
        return pickle.loads(base64.b64decode(base64_str.encode('utf-8')))

    def _get_model_for_dataset(self, dataset_name: str):
        return DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))

    def reload_data_if_needed(self, dataset_name: str,
                               distribution: str = None,
                               verbose: bool = False) -> bool:
        needs_reload = (
            dataset_name != self.current_dataset_name or
            (distribution is not None and distribution != self.current_distribution)
        )

        print(f"DEBUG reload_data_if_needed: needs_reload={needs_reload} | "
          f"current={self.current_dataset_name} vs server={dataset_name}")
        
        if not needs_reload:
            return False

        if dataset_name != self.current_dataset_name:
            model_name, num_classes, in_channels, input_size = \
                self._get_model_for_dataset(dataset_name)

            print(f"Client {self.client_id}: 🔄 Dataset change: "
                  f"{self.current_dataset_name} → {dataset_name} | "
                  f"model={model_name} | in_channels={in_channels} | "
                  f"input_size={input_size} | num_classes={num_classes}")

            self.model = create_model(model_name, num_classes=num_classes,
                                       in_channels=in_channels, input_size=input_size)

            self.trainer.model = self.model
            if self.trainer.optimizer_name.lower() == "sgd":
                self.trainer.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.trainer.learning_rate,
                    momentum=self.trainer.momentum,
                    weight_decay=self.trainer.weight_decay
                    )
            elif self.trainer.optimizer_name.lower() == "adam":
                self.trainer.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.trainer.learning_rate,
                    weight_decay=self.trainer.weight_decay
                    )


            self.current_dataset_name = dataset_name
            self.current_model_name = model_name

        if distribution is not None and distribution != self.current_distribution:
            self.distribution = distribution
            self.current_distribution = distribution

        print(f"Client {self.client_id}: 📦 Loading data for "
              f"{self.current_dataset_name} ({self.distribution})...")

        self.train_loader = create_single_client_loader(
            dataset_name=self.current_dataset_name,
            client_id=self.client_id,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            distribution=self.distribution,
            data_dir=self.data_dir,
            seed=42
        )

        print(f"Client {self.client_id}: ✅ Data reloaded "
              f"({len(self.train_loader.dataset)} samples)")

        self._cleanup_memory()
        return True

    def participate_in_round(self, verbose: bool = False) -> bool:
        try:
            # Step 1: Get global model
            response = requests.get(
                f"{self.server_url}/api/get_global_model", timeout=10
            )
            if response.status_code != 200:
                print(f"Client {self.client_id}: ❌ get_global_model "
                      f"returned {response.status_code}")
                return False

            data = response.json()
            weights = self._base64_to_weights(data['weights'])
            current_round = data['round']
            xfl_strategy = data.get('xfl_strategy', 'all_layers')
            xfl_param = data.get('xfl_param', 3)
            server_dataset = data.get('dataset_name', 'MNIST')
            server_distribution = data.get('data_distribution', 'iid')

            
            local_state = self.trainer.model.state_dict()
            debug_layers = ['conv1.weight', 'conv2.weight', 'fc1.weight']
            for layer in debug_layers:
                if layer in weights:
                    server_shape = weights[layer].shape
                    if layer in local_state:
                        local_shape = local_state[layer].shape
                        if server_shape != local_shape:
                            print(f"Client {self.client_id}: ⚠️  {layer} "
                                  f"expected {local_shape} got {server_shape}")
                    else:
                        print(f"Client {self.client_id}: ⚠️  {layer} "
                              f"({server_shape}) missing in local model")
                elif layer in local_state:
                    print(f"Client {self.client_id}: ⚠️  {layer} "
                          f"({local_state[layer].shape}) missing from server")

            # Step 2: Reload model/data if dataset changed
            self.reload_data_if_needed(server_dataset, server_distribution, verbose)

            # DEBUG client loader batch shape
            try:
                first_batch = next(iter(self.train_loader))
                print(f"DEBUG client.py Client{self.client_id} first batch data.shape: {first_batch[0].shape}")
            except Exception as e:
                print(f"DEBUG client.py Client{self.client_id} loader debug error: {e}")

            if verbose:
                print(f"Client {self.client_id}: Round {current_round} | "
                      f"dataset={server_dataset} | strategy={xfl_strategy}")

# Step 3: Handle XFL partial updates (server sends subset of layers)
            local_keys = set(self.trainer.model.state_dict().keys())
            server_keys = set(weights.keys())
            
            # Log XFL partial updates
            if verbose or len(server_keys) < len(local_keys):
                num_server_layers = len(set(k.split('.')[0] for k in server_keys))
                num_local_layers = len(local_keys)
                print(f"Client {self.client_id}: ℹ️ XFL partial update — "
                      f"server sent {len(server_keys)}/{num_local_layers} params "
                      f"({num_server_layers} layers)")


            # Step 4: Load server weights
            try:
                self.trainer.set_model_weights(weights)
            except Exception as e:
                print(f"Client {self.client_id}: ❌ set_model_weights failed: {e}")
                return False

            # Step 5: Train locally
            start_train = time.time()
            training_metrics = self.trainer.train(
                train_loader=self.train_loader,
                num_epochs=self.local_epochs,
                verbose=False
            )
            train_time = time.time() - start_train

            if verbose:
                print(f"Client {self.client_id}: Training done in {train_time:.1f}s | "
                      f"Loss={training_metrics['loss']:.4f} | "
                      f"Acc={training_metrics['accuracy']:.1f}%")

            # Step 6: Apply XFL strategy
            trained_weights = self.trainer.get_model_weights()
            num_samples = len(self.train_loader.dataset)

            weights_to_send = self._apply_xfl_strategy(
                trained_weights, xfl_strategy, xfl_param, current_round, verbose
            )

# Step 7: Prepare payload (metrics collected AFTER transmission)
            model_size_bytes = sum(
                p.numel() * p.element_size()
                for p in weights_to_send.values()
                if hasattr(p, 'element_size')
            )

            payload = {
                "client_id": self.client_id,
                "weights": self._weights_to_base64(weights_to_send),
                "num_samples": num_samples,
                "metrics": {}  # Placeholder - will be filled after transmission
            }

            if hasattr(self, '_quantization_meta') and self._quantization_meta:
                payload["quantization_meta"] = self._quantization_meta
                delattr(self, '_quantization_meta')

            # MEASURE REAL TRANSMISSION ↓
            start_tx = time.perf_counter()
            response = requests.post(
                f"{self.server_url}/api/submit_update",
                json=payload,
                timeout=30
            )
            tx_time = time.perf_counter() - start_tx
            tx_latency_ms = tx_time * 1000  # Real end-to-end latency

            if response.status_code != 200:
                print(f"Client {self.client_id}: ❌ submit_update returned "
                      f"{response.status_code}: {response.text[:200]}")
                return False

            result = response.json()

            # NOW collect REAL metrics after successful transmission
            network_metrics = self.metrics_collector.get_network_metrics(
                bytes_sent=model_size_bytes,
                bytes_received=model_size_bytes,
                transmission_time=tx_time
            )
            network_metrics.update({
                "http_status": response.status_code,
                "is_real_measurement": True
            })

            full_metrics = self.metrics_collector.collect_full_metrics(
                training_metrics=training_metrics,
                model_weights=weights_to_send,
                network_metrics=network_metrics
            )

            # Store the REAL complete metrics to DB via server
            payload_with_metrics = {
                "client_id": self.client_id,
                "weights": self._weights_to_base64(weights_to_send),
                "num_samples": num_samples,
                "metrics": full_metrics  # Now includes real transmission_time_sec, latency_ms
            }

            if hasattr(self, '_quantization_meta') and self._quantization_meta:
                payload_with_metrics["quantization_meta"] = self._quantization_meta
                delattr(self, '_quantization_meta')

            # Send FINAL payload with REAL metrics to server
            final_response = requests.post(
                f"{self.server_url}/api/submit_update",
                json=payload_with_metrics,
                timeout=30
            )

            if final_response.status_code == 200:
                result = final_response.json()
                if verbose:
                    print(f"Client {self.client_id}: ✅ Final metrics sent | "
                          f"{result.get('submissions')} total submissions")
            else:
                print(f"Client {self.client_id}: ⚠️ Final metrics send failed "
                      f"(status={final_response.status_code}) but model already sent")

            if verbose:
                print(f"Client {self.client_id}: ✅ Submitted | "
                      f"{result.get('submissions')} updates received by server | "
                      f"Real tx_time={tx_time:.3f}s | latency={tx_latency_ms:.1f}ms")

            self._cleanup_memory()
            return True
            payload = {
                "client_id": self.client_id,
                "weights": self._weights_to_base64(weights_to_send),
                "num_samples": num_samples,
                "metrics": full_metrics
            }

            if hasattr(self, '_quantization_meta') and self._quantization_meta:
                payload["quantization_meta"] = self._quantization_meta
                delattr(self, '_quantization_meta')

            response = requests.post(
                f"{self.server_url}/api/submit_update",
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Client {self.client_id}: ❌ submit_update returned "
                      f"{response.status_code}: {response.text[:200]}")
                return False

            result = response.json()

            if verbose:
                print(f"Client {self.client_id}: ✅ Submitted | "
                      f"{result.get('submissions')} updates received by server")

            self._cleanup_memory()
            return True

        except requests.exceptions.Timeout as e:
            print(f"Client {self.client_id}: ❌ Timeout error: {e}")
            self._cleanup_memory()
            return False
        except Exception as e:
            import traceback
            print(f"Client {self.client_id}: ❌ Unexpected error in "
                  f"participate_in_round:")
            print(traceback.format_exc())
            self._cleanup_memory()
            return False

    def _apply_sparsification(self, weights: OrderedDict, threshold: float) -> OrderedDict:
        return OrderedDict(
            (name, torch.where(torch.abs(param) < threshold,
                               torch.zeros_like(param), param))
            for name, param in weights.items()
        )

    def _apply_quantization(self, weights: OrderedDict,
                             bits: int) -> Tuple[OrderedDict, Dict[str, Dict]]:
        quantized = OrderedDict()
        quantization_meta = {}

        for name, param in weights.items():
            min_val = param.min().item()
            max_val = param.max().item()

            if max_val == min_val:
                quantized[name] = param
                quantization_meta[name] = {
                    'quantized': False, 'min_val': min_val, 'max_val': max_val,
                    'scale': 1.0, 'zero_point': 0
                }
                continue

            abs_max = max(abs(min_val), abs(max_val))
            scale = (2 * abs_max) / (2**bits - 1)

            quantized_int = torch.round(param / scale).clamp(
                -(2**(bits-1)), 2**(bits-1) - 1
            ).to(torch.int32)
            quantized[name] = quantized_int
            quantization_meta[name] = {
                'quantized': True, 'min_val': min_val, 'max_val': max_val,
                'scale': scale, 'zero_point': 0, 'bits': bits
            }

        return quantized, quantization_meta

    def _apply_xfl_strategy(self, trained_weights: OrderedDict, xfl_strategy: str,
                              xfl_param: int, current_round: int,
                              verbose: bool = False) -> OrderedDict:
        all_layer_names = list(trained_weights.keys())
        
        # FIXED: xfl_sparsification sends FIRST 2 conv blocks only (8 params: conv1/2 + bias)
        if xfl_strategy == "xfl_sparsification":
            # Target layers: conv1.weight, conv1.bias, conv2.weight, conv2.bias (4 params ×2 = 8)
            target_layers = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias']
            selected_weights = OrderedDict(
                (name, trained_weights[name]) for name in target_layers if name in trained_weights
            )
            
            # NEW: Log sent shapes
            print(f"Client {self.client_id}: 📤 Sending XFL conv1/2:")
            for name in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias']:
                if name in selected_weights:
                    print(f"  {name}: {selected_weights[name].shape}")
            
            # Apply sparsification
            selected_weights = self._apply_sparsification(
                selected_weights, self.sparsification_threshold
            )
            
            if verbose:
                print(f"   📌 XFL-Sparsification: conv1/conv2 only ({len(selected_weights)}/8 params)")
            return selected_weights


        if xfl_strategy == "all_layers":
            return trained_weights

        elif xfl_strategy in ["first_n_layers", "last_n_layers", "random_layers"]:
            return trained_weights

        elif xfl_strategy.startswith("xfl"):
            # Generic cyclic XFL (other variants)
            layer_prefixes = sorted(set(name.split('.')[0] for name in all_layer_names))
            num_layers = len(layer_prefixes)
            layer_index = (self.client_id + current_round - 1) % num_layers
            selected_prefix = layer_prefixes[layer_index]

            selected_weights = OrderedDict(
                (param_name, trained_weights[param_name])
                for param_name in all_layer_names
                if param_name.startswith(selected_prefix)
            )

            if verbose:
                print(f"   📌 XFL-{xfl_strategy}: layer {layer_index} ({selected_prefix}) — "
                      f"{len(selected_weights)} params")

            if "sparsification" in xfl_strategy:
                selected_weights = self._apply_sparsification(
                    selected_weights, self.sparsification_threshold
                )
            elif "quantization" in xfl_strategy:
                selected_weights, quantization_meta = self._apply_quantization(
                    selected_weights, self.quantization_bits
                )
                self._quantization_meta = quantization_meta

            return selected_weights

        else:
            if verbose:
                print(f"   ⚠️ Unknown XFL strategy: {xfl_strategy}, sending all layers")
            return trained_weights

