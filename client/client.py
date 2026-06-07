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
import random
from typing import Dict, Any, Tuple
import torch
import torch.optim as optim

from .trainer import LocalTrainer
from .metrics import MetricsCollector
from .model import create_model, DATASET_CONFIG, get_model_params_for_dataset
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
        data_dir: str = "./data",
        use_real_hardware_metrics: bool = False,
        real_metrics_collector = None
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.timeout = timeout
        self.max_retries = 5
        self.base_delay = 1

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer

        self.sparsification_threshold = sparsification_threshold
        self.quantization_bits = quantization_bits

        # Real hardware metrics collector (for Raspberry Pi)
        self.use_real_hardware_metrics = use_real_hardware_metrics
        self.real_metrics_collector = real_metrics_collector

        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.data_dir = data_dir
        self.current_dataset_name = dataset_name
        self.current_model_name = model.__class__.__name__
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

    def _get_resource_usage(self) -> Dict[str, float]:
        if not hasattr(self, 'metrics_collector') or self.metrics_collector is None:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_percent': 0.0
            }

        metrics = self.metrics_collector.get_system_metrics()
        return {
            'cpu_percent': metrics.get('system_cpu_percent', 0.0),
            'memory_mb': metrics.get('process_memory_mb', 0.0),
            'memory_percent': metrics.get('system_memory_percent', 0.0)
        }

    def reload_data_if_needed(self, dataset_name: str,
                               model_name: str = None,
                               distribution: str = None,
                               num_clients: int = None,
                               batch_size: int = None,
                               verbose: bool = False) -> bool:
        needs_reload = (
            dataset_name != self.current_dataset_name or
            (model_name is not None and model_name != self.current_model_name) or
            (distribution is not None and distribution != self.current_distribution) or
            (num_clients is not None and num_clients != self.num_clients) or
            (batch_size is not None and batch_size != self.batch_size)
        )

        print(f"DEBUG reload_data_if_needed: needs_reload={needs_reload} | "
          f"current={self.current_dataset_name} vs server={dataset_name} | "
          f"num_clients {self.num_clients} vs {num_clients} | "
          f"batch_size {self.batch_size} vs {batch_size}")
        
        if not needs_reload:
            return False

        # Determine which model should be used for the current dataset.
        target_model_name = model_name if model_name is not None else self._get_model_for_dataset(dataset_name)[0]
        
        # Get the correct parameters for the target model and dataset
        num_classes, in_channels, input_size = get_model_params_for_dataset(target_model_name, dataset_name)

        if dataset_name != self.current_dataset_name or target_model_name != self.current_model_name:
            print(f"Client {self.client_id}: 🔄 Model or dataset change: "
                  f"{self.current_dataset_name}/{self.current_model_name} → {dataset_name}/{target_model_name} | "
                  f"in_channels={in_channels} | input_size={input_size} | num_classes={num_classes}")

            self.model = create_model(target_model_name, num_classes=num_classes,
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
            self.current_model_name = target_model_name

        if distribution is not None and distribution != self.current_distribution:
            self.distribution = distribution
            self.current_distribution = distribution

        if num_clients is not None and num_clients != self.num_clients:
            self.num_clients = num_clients

        if batch_size is not None and batch_size != self.batch_size:
            self.batch_size = batch_size

        print(f"Client {self.client_id}: 📦 Loading data for "
              f"{self.current_dataset_name} ({self.distribution}) with {self.num_clients} clients, "
              f"batch size={self.batch_size}...")

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

    def _simulate_network_delay(self, size_bytes: int, direction: str = 'upload') -> float:
        """Estimate a simulated network delay for upload/download."""
        if not hasattr(self, 'metrics_collector') or not self.metrics_collector:
            return 0.0

        network_config = self.metrics_collector.network_config
        if not network_config.get('simulate_constraints', False):
            return 0.0

        bandwidth_mbps = max(0.1, float(network_config.get('bandwidth_mbps', 10)))
        latency_ms = float(network_config.get('latency_ms', 0))
        latency_std_ms = float(network_config.get('latency_std_ms', 0))
        packet_loss_rate = float(network_config.get('packet_loss_rate', 0.0))

        simulated_latency = max(0.0, random.gauss(latency_ms, latency_std_ms)) / 1000.0
        transfer_time = (size_bytes * 8) / (bandwidth_mbps * 1_000_000)
        delay = simulated_latency + transfer_time

        if packet_loss_rate > 0 and random.random() < packet_loss_rate:
            extra_loss_delay = min(2.0, transfer_time + simulated_latency)
            print(f"Client {self.client_id}: ⚠️ Simulated {direction} packet loss, adding {extra_loss_delay:.2f}s delay")
            delay += extra_loss_delay

        return max(0.0, delay)

    def participate_in_round(self, verbose: bool = False) -> bool:
        try:
            # NOTE: reset_round_metrics() is now called in run_client_raspberry_pi.py
            # before this function is called - don't call it here to avoid double-reset
            
            # Step 1: Get global model
            response = requests.get(
                f"{self.server_url}/api/get_global_model?client_id={self.client_id}", timeout=10
            )
            if response.status_code != 200:

                print(f"Client {self.client_id}: ❌ get_global_model "
                      f"returned {response.status_code}")
                return False


            data = response.json()
            download_delay = self._simulate_network_delay(len(response.content), 'download')
            if download_delay > 0:
                if verbose:
                    print(f"Client {self.client_id}: ⏳ Simulated download delay {download_delay:.2f}s")
                time.sleep(download_delay)

            weights = self._base64_to_weights(data['weights'])
            current_round = data['round']
            xfl_strategy = data.get('xfl_strategy', 'all_layers')
            xfl_param = data.get('xfl_param', 3)
            server_dataset = data.get('dataset_name', 'MNIST')
            server_distribution = data.get('data_distribution', 'iid')
            server_num_clients = data.get('num_clients', self.num_clients)
            server_batch_size = data.get('batch_size', self.batch_size)
            server_local_epochs = data.get('local_epochs', self.local_epochs)
            server_learning_rate = data.get('learning_rate', self.learning_rate)
            server_model_name = data.get('model_name', self.current_model_name)

            # Update local training hyperparameters if changed
            if server_local_epochs != self.local_epochs:
                print(f"Client {self.client_id}: 🔄 local_epochs changed: {self.local_epochs} → {server_local_epochs}")
                self.local_epochs = int(server_local_epochs)

            if server_learning_rate != self.learning_rate:
                print(f"Client {self.client_id}: 🔄 learning_rate changed: {self.learning_rate} → {server_learning_rate}")
                self.learning_rate = float(server_learning_rate)
                self.trainer.learning_rate = self.learning_rate
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Update network config if changed
            raw_simulate_constraints = data.get('simulate_constraints', False)
            if isinstance(raw_simulate_constraints, str):
                simulate_constraints = raw_simulate_constraints.lower() in ('true', '1', 'yes')
            else:
                simulate_constraints = bool(raw_simulate_constraints)

            server_network_config = {
                'latency_ms': data.get('network_latency_ms', 0),
                'latency_std_ms': data.get('network_latency_std_ms', 0),
                'bandwidth_mbps': data.get('network_bandwidth_mbps', 10),
                'packet_loss_rate': data.get('network_packet_loss_rate', 0),
                'jitter_ms': data.get('jitter_ms', 0),
                'simulate_constraints': simulate_constraints
            }
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.update_network_config(server_network_config)

            # Check system limits (CPU, RAM) and add to metrics if exceeded
            server_cpu_limit = data.get('cpu_limit', 100)
            server_ram_limit = data.get('ram_limit', 2048)
            cpu_limit_exceeded = False
            ram_limit_exceeded = False
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                current_metrics = self.metrics_collector.get_system_metrics()
                current_cpu = current_metrics.get('system_cpu_percent', 0)
                current_ram_mb = current_metrics.get('process_memory_mb', 0)

                # Add limit info to metrics for server-side monitoring
                self.metrics_collector.network_config['cpu_limit'] = server_cpu_limit
                self.metrics_collector.network_config['ram_limit'] = server_ram_limit

                # No blocking: just notify when limits are exceeded
                cpu_limit_exceeded = current_cpu > server_cpu_limit
                ram_limit_exceeded = current_ram_mb > server_ram_limit

                if cpu_limit_exceeded:
                    print(f"⚠️ Client {self.client_id}: CPU limit exceeded! "
                          f"Current: {current_cpu}% > Limit: {server_cpu_limit}%")
                if ram_limit_exceeded:
                    print(f"⚠️ Client {self.client_id}: RAM limit exceeded! "
                          f"Current: {current_ram_mb:.1f}MB > Limit: {server_ram_limit}MB")

            # Update num_clients if changed
            if server_num_clients != self.num_clients:
                print(f"Client {self.client_id}: 🔄 num_clients changed: {self.num_clients} → {server_num_clients}")
                self.num_clients = int(server_num_clients)

            if server_batch_size != self.batch_size:
                print(f"Client {self.client_id}: 🔄 batch_size changed: {self.batch_size} → {server_batch_size}")
                self.batch_size = int(server_batch_size)

            self.reload_data_if_needed(
                server_dataset,
                model_name=server_model_name,
                distribution=server_distribution,
                num_clients=server_num_clients,
                batch_size=server_batch_size,
                verbose=verbose
            )

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

# Step 7: Prepare payload and send weights + real metrics in a single request
            model_size_bytes = sum(
                p.numel() * p.element_size()
                for p in weights_to_send.values()
                if hasattr(p, 'element_size')
            )

            # Build metrics now so the server receives a real client submission immediately
            # Measure current resource usage for limit detection
            current_metrics = self.metrics_collector.get_system_metrics() if hasattr(self, 'metrics_collector') else {}
            current_cpu_percent = current_metrics.get('system_cpu_percent', 0)
            current_memory_mb = current_metrics.get('process_memory_mb', 0)

            if self.use_real_hardware_metrics and self.real_metrics_collector is not None:
                real_summary = self.real_metrics_collector.get_summary()
                metrics_payload = {
                    'training_loss': training_metrics.get('loss', 0),
                    'training_accuracy': training_metrics.get('accuracy', 0),
                    'training_time_sec': train_time,
                    'num_samples': num_samples,
                    'bytes_sent': model_size_bytes,
                    'bytes_received': model_size_bytes,
                    'cpu_percent': real_summary.get('cpu', {}).get('avg_percent', current_cpu_percent),
                    'memory_mb': real_summary.get('memory', {}).get('avg_mb', current_memory_mb),
                    'temperature_celsius': real_summary.get('temperature', {}).get('avg_celsius'),
                    'energy_wh': real_summary.get('energy', {}).get('total_wh', 0),
                    'latency_ms': real_summary.get('network', {}).get('avg_latency_ms', 0),
                    'jitter_ms': real_summary.get('network', {}).get('avg_jitter_ms', 0),
                    'packet_loss_rate': real_summary.get('network', {}).get('packet_loss_rate', 0),
                    'throughput_mbps': real_summary.get('network', {}).get('avg_bandwidth_mbps', 0),
                    'cpu_limit': server_cpu_limit,
                    'ram_limit': server_ram_limit,
                    'cpu_limit_exceeded': cpu_limit_exceeded,
                    'ram_limit_exceeded': ram_limit_exceeded,
                    'is_real_measurement': True
                }
            else:
                metrics_payload = {
                    'training_loss': training_metrics.get('loss', 0),
                    'training_accuracy': training_metrics.get('accuracy', 0),
                    'training_time_sec': train_time,
                    'num_samples': num_samples,
                    'bytes_sent': model_size_bytes,
                    'bytes_received': model_size_bytes,
                    'cpu_percent': current_cpu_percent,
                    'memory_mb': current_memory_mb,
                    'cpu_limit': server_cpu_limit,
                    'ram_limit': server_ram_limit,
                    'cpu_limit_exceeded': cpu_limit_exceeded,
                    'ram_limit_exceeded': ram_limit_exceeded,
                    'is_real_measurement': True
                }

            placeholder_payload = {
                "client_id": self.client_id,
                "weights": self._weights_to_base64(weights_to_send),
                "num_samples": num_samples,
                "metrics": {}
            }

            if hasattr(self, '_quantization_meta') and self._quantization_meta:
                placeholder_payload["quantization_meta"] = self._quantization_meta
                delattr(self, '_quantization_meta')

            tx_timeout = 180
            print(f"🔍 DEBUG: upload submit_update timeout={tx_timeout}s (real_hardware={self.use_real_hardware_metrics})")
            upload_delay = self._simulate_network_delay(model_size_bytes, 'upload')
            if upload_delay > 0 and verbose:
                print(f"Client {self.client_id}: ⏳ Simulated upload delay {upload_delay:.2f}s")
            time.sleep(upload_delay)
            start_tx = time.perf_counter()

            max_retries = 3
            retry_delay = 5
            response = None
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.server_url}/api/submit_update",
                        json=placeholder_payload,
                        timeout=tx_timeout
                    )
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                        print("   ⏳ Retrying upload in 5s...")
                        time.sleep(retry_delay)
                    else:
                        print(f"   ❌ Upload failed after {max_retries} attempts")

            if response is None:
                print(f"Client {self.client_id}: ❌ submit_update upload failed: {last_error}")
                self._cleanup_memory()
                return False

            wire_tx_time = time.perf_counter() - start_tx
            total_tx_time = wire_tx_time + upload_delay
            tx_latency_ms = total_tx_time * 1000  # Effective end-to-end latency
            if model_size_bytes and total_tx_time > 0:
                measured_throughput_mbps = (model_size_bytes * 8) / (total_tx_time * 1_000_000)
            else:
                measured_throughput_mbps = 0.0

            # After upload, send the real metrics update so the server stores the true transmission time and network metrics
            network_metrics = self.metrics_collector.get_network_metrics(
                bytes_sent=model_size_bytes,
                bytes_received=model_size_bytes,
                transmission_time=total_tx_time
            ) if hasattr(self, 'metrics_collector') and self.metrics_collector else {
                'latency_ms': tx_latency_ms,
                'packet_loss_rate': 0.0,
                'jitter_ms': 0.0,
                'transmission_time_sec': total_tx_time,
                'throughput_mbps': round(measured_throughput_mbps, 6),
                'bytes_sent': model_size_bytes,
                'bytes_received': model_size_bytes
            }

            metrics_payload['transmission_time_sec'] = network_metrics.get('transmission_time_sec', total_tx_time)
            metrics_payload['throughput_mbps'] = network_metrics.get('throughput_mbps', round(measured_throughput_mbps, 6))
            metrics_payload['latency_ms'] = network_metrics.get('latency_ms', tx_latency_ms)
            metrics_payload['packet_loss_rate'] = network_metrics.get('packet_loss_rate', 0.0)
            metrics_payload['jitter_ms'] = network_metrics.get('jitter_ms', 0.0)
            metrics_payload['bytes_sent'] = network_metrics.get('bytes_sent', model_size_bytes)
            metrics_payload['bytes_received'] = network_metrics.get('bytes_received', model_size_bytes)

            metrics_update_payload = {
                "client_id": self.client_id,
                "num_samples": num_samples,
                "metrics": metrics_payload
            }

            update_response = None
            update_error = None
            for attempt in range(max_retries):
                try:
                    update_response = requests.post(
                        f"{self.server_url}/api/submit_update",
                        json=metrics_update_payload,
                        timeout=tx_timeout
                    )
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    update_error = e
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ Metrics update attempt {attempt + 1}/{max_retries} failed: {e}")
                        print("   ⏳ Retrying metrics update in 5s...")
                        time.sleep(retry_delay)
                    else:
                        print(f"   ❌ Metrics update failed after {max_retries} attempts")

            if update_response is None:
                print(f"Client {self.client_id}: ❌ metrics submit_update failed: {update_error}")
                self._cleanup_memory()
                return False

            if response.status_code != 200:
                print(f"Client {self.client_id}: ❌ initial submit_update returned "
                      f"{response.status_code}: {response.text[:200]}")
                return False

            if update_response.status_code != 200:
                print(f"Client {self.client_id}: ❌ metrics submit_update returned "
                      f"{update_response.status_code}: {update_response.text[:200]}")
                return False

            result = update_response.json()

            if verbose:
                print(f"Client {self.client_id}: ✅ Submitted | "
                      f"{result.get('submissions')} updates received by server | "
                      f"Real tx_time={total_tx_time:.3f}s | latency={tx_latency_ms:.1f}ms | "
                      f"throughput={metrics_payload['throughput_mbps']:.3f} Mbps")

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

