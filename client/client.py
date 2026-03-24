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
from .model import create_model, DATASET_CONFIG
from .dataset import create_dataloaders

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

        self.metrics_collector = MetricsCollector(client_id=client_id)
        # Démarrer la collecte de métriques dès l'initialisation
        self.metrics_collector.start_collection()

        print(f"✅ FLClient {client_id} initialized | dataset={dataset_name} | "
              f"model={self.current_model_name} | {len(train_loader.dataset)} samples")

    def _cleanup_memory(self):
        """Clean up memory after training to prevent OOM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if hasattr(self, '_quantization_meta'):
            delattr(self, '_quantization_meta')

    def _weights_to_base64(self, weights: OrderedDict) -> str:
        weights_bytes = pickle.dumps(weights)
        return base64.b64encode(weights_bytes).decode('utf-8')

    def _base64_to_weights(self, base64_str: str) -> OrderedDict:
        weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
        return pickle.loads(weights_bytes)

    def _get_model_for_dataset(self, dataset_name: str):
        """Retourne (model_name, num_classes, in_channels, input_size)"""
        return DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))

    def reload_data_if_needed(self, dataset_name: str, distribution: str = None, verbose: bool = False) -> bool:
        """Reload model and data if dataset or distribution changed."""
        needs_reload = (
            dataset_name != self.current_dataset_name or
            (distribution is not None and distribution != self.current_distribution)
        )

        if not needs_reload:
            return False

        if dataset_name != self.current_dataset_name:
            model_name, num_classes, in_channels, input_size = self._get_model_for_dataset(dataset_name)

            print(f"Client {self.client_id}: 🔄 Dataset change detected: "
                  f"{self.current_dataset_name} → {dataset_name}")
            print(f"Client {self.client_id}: 🔄 New model: {model_name} | "
                  f"in_channels={in_channels} | input_size={input_size} | num_classes={num_classes}")

            self.model = create_model(
                model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                input_size=input_size
            )

            self.trainer = LocalTrainer(
                model=self.model,
                optimizer_name="sgd",
                learning_rate=0.01,
                momentum=0.9,
                weight_decay=0.0001
            )

            self.current_dataset_name = dataset_name
            self.current_model_name = model_name

        if distribution is not None and distribution != self.current_distribution:
            self.distribution = distribution
            self.current_distribution = distribution

        print(f"Client {self.client_id}: 📦 Loading data for {self.current_dataset_name} "
              f"(distribution={self.distribution})...")

        client_loaders, _ = create_dataloaders(
            dataset_name=self.current_dataset_name,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            distribution=self.distribution,
            data_dir=self.data_dir,
            seed=42
        )

        self.train_loader = client_loaders[self.client_id]

        print(f"Client {self.client_id}: ✅ Data reloaded "
              f"({len(self.train_loader.dataset)} samples)")

        self._cleanup_memory()
        return True

    def participate_in_round(self, verbose: bool = False) -> bool:
        """
        Participate in one FL round.
        verbose=True affiche TOUTES les erreurs pour faciliter le debug.
        """
        try:
            # ── Step 1: Get global model ──────────────────────────────────────
            response = requests.get(
                f"{self.server_url}/api/get_global_model",
                timeout=10
            )
            if response.status_code != 200:
                print(f"Client {self.client_id}: ❌ get_global_model returned {response.status_code}")
                return False

            data = response.json()
            weights = self._base64_to_weights(data['weights'])
            current_round = data['round']
            xfl_strategy = data.get('xfl_strategy', 'all_layers')
            xfl_param = data.get('xfl_param', 3)

            server_dataset = data.get('dataset_name', 'MNIST')
            server_distribution = data.get('data_distribution', 'iid')

            # ── Step 2: Reload model/data if dataset changed ──────────────────
            # DOIT être fait avant set_model_weights
            self.reload_data_if_needed(server_dataset, server_distribution, verbose)

            if verbose:
                print(f"Client {self.client_id}: Round {current_round} | "
                      f"dataset={server_dataset} | strategy={xfl_strategy}")

            # ── Step 3: Validate that server weights match local model ────────
            local_keys = set(self.trainer.model.state_dict().keys())
            server_keys = set(weights.keys())

            if local_keys != server_keys:
                print(f"Client {self.client_id}: ❌ Weight key mismatch AFTER reload!")
                print(f"  Local model keys  : {sorted(local_keys)}")
                print(f"  Server weight keys: {sorted(server_keys)}")
                print(f"  Missing in local  : {server_keys - local_keys}")
                print(f"  Extra in local    : {local_keys - server_keys}")
                return False

            # ── Step 4: Load server weights into local model ──────────────────
            try:
                self.trainer.set_model_weights(weights)
            except Exception as e:
                print(f"Client {self.client_id}: ❌ set_model_weights failed: {e}")
                return False

            # ── Step 5: Train locally ─────────────────────────────────────────
            start_train = time.time()
            training_metrics = self.trainer.train(
                train_loader=self.train_loader,
                num_epochs=self.local_epochs,
                verbose=False
            )
            train_time = time.time() - start_train

            if verbose:
                print(f"Client {self.client_id}: Training done in {train_time:.1f}s | "
                      f"Loss={training_metrics['loss']:.4f} | Acc={training_metrics['accuracy']:.1f}%")

            # ── Step 6: Get trained weights and apply XFL strategy ────────────
            trained_weights = self.trainer.get_model_weights()
            num_samples = len(self.train_loader.dataset)

            weights_to_send = self._apply_xfl_strategy(
                trained_weights, xfl_strategy, xfl_param, current_round, verbose
            )

            # ── Step 7: Collect metrics ───────────────────────────────────────
            model_size_bytes = sum(
                p.numel() * p.element_size()
                for p in weights_to_send.values()
                if hasattr(p, 'element_size')
            )

            import random
            latency_ms = random.uniform(5, 50)
            packet_loss_rate = random.uniform(0, 0.05)
            jitter_ms = random.uniform(1, 10)

            full_metrics = self.metrics_collector.collect_full_metrics(
                training_metrics=training_metrics,
                model_weights=weights_to_send,
                network_metrics={
                    "bytes_sent": model_size_bytes,
                    "bytes_received": model_size_bytes,
                    "transmission_time": 0,
                    "latency_ms": latency_ms,
                    "packet_loss_rate": packet_loss_rate,
                    "jitter_ms": jitter_ms
                }
            )

            # ── Step 8: Send update to server ─────────────────────────────────
            weights_b64 = self._weights_to_base64(weights_to_send)

            payload = {
                "client_id": self.client_id,
                "weights": weights_b64,
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
                print(f"Client {self.client_id}: ❌ submit_update returned {response.status_code}: "
                      f"{response.text[:200]}")
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
            # Toujours afficher l'exception complète pour faciliter le debug
            import traceback
            print(f"Client {self.client_id}: ❌ Unexpected error in participate_in_round:")
            print(traceback.format_exc())
            self._cleanup_memory()
            return False

    def _apply_sparsification(self, weights: OrderedDict, threshold: float) -> OrderedDict:
        sparsified = OrderedDict()
        for name, param in weights.items():
            sparsified[name] = torch.where(
                torch.abs(param) < threshold,
                torch.zeros_like(param),
                param
            )
        return sparsified

    def _apply_quantization(self, weights: OrderedDict, bits: int) -> Tuple[OrderedDict, Dict[str, Dict]]:
        quantized = OrderedDict()
        quantization_meta = {}

        for name, param in weights.items():
            min_val = param.min().item()
            max_val = param.max().item()

            if max_val == min_val:
                quantized[name] = param
                quantization_meta[name] = {
                    'quantized': False,
                    'min_val': min_val,
                    'max_val': max_val,
                    'scale': 1.0,
                    'zero_point': 0
                }
                continue

            abs_max = max(abs(min_val), abs(max_val))
            scale = (2 * abs_max) / (2**bits - 1)
            zero_point = 0

            quantized_int = torch.round(param / scale).clamp(
                -(2**(bits-1)), 2**(bits-1) - 1
            ).to(torch.int32)
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
        all_layer_names = list(trained_weights.keys())

        if xfl_strategy == "all_layers":
            return trained_weights

        elif xfl_strategy in ["first_n_layers", "last_n_layers", "random_layers"]:
            return trained_weights

        elif xfl_strategy.startswith("xfl"):
            layer_prefixes = sorted(set(name.split('.')[0] for name in all_layer_names))
            num_layers = len(layer_prefixes)
            layer_index = (self.client_id + current_round - 1) % num_layers
            selected_prefix = layer_prefixes[layer_index]

            selected_weights = OrderedDict()
            for param_name in all_layer_names:
                if param_name.startswith(selected_prefix + '.'):
                    selected_weights[param_name] = trained_weights[param_name]

            if verbose:
                print(f"   📌 XFL: layer {layer_index} ({selected_prefix}) — "
                      f"{len(selected_weights)} params")

            if xfl_strategy == "xfl_sparsification":
                selected_weights = self._apply_sparsification(
                    selected_weights, self.sparsification_threshold
                )
            elif xfl_strategy == "xfl_quantization":
                selected_weights, quantization_meta = self._apply_quantization(
                    selected_weights, self.quantization_bits
                )
                self._quantization_meta = quantization_meta

            return selected_weights

        else:
            if verbose:
                print(f"   ⚠️ Unknown XFL strategy: {xfl_strategy}, sending all layers")
            return trained_weights