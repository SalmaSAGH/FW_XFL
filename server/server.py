"""
Flask server for Federated Learning with XFL support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import base64
from collections import OrderedDict
from typing import Dict, List, Any
import time
import threading

from .strategy import create_aggregation_strategy, XFL, FedAvg
from .metrics import ServerMetricsCollector


class FLServer:
    """
    Federated Learning Server with XFL support
    """
    
    def __init__(
        self,
        global_model: torch.nn.Module,
        aggregation_strategy: str = "fedavg",
        num_rounds: int = 10,
        clients_per_round: int = 5,
        db_path: str = "logs/server_metrics.db",
        xfl_strategy: str = "all_layers",
        xfl_param: int = 3
    ):
        """
        Args:
            global_model: Initial global model
            aggregation_strategy: Aggregation method ('fedavg', 'fedprox')
            num_rounds: Total number of FL rounds
            clients_per_round: Number of clients per round
            db_path: Path to metrics database
            xfl_strategy: XFL layer selection strategy
            xfl_param: XFL parameter (e.g., N for first_n/last_n)
        """
        self.global_model = global_model
        self.aggregation_strategy = create_aggregation_strategy(
            aggregation_strategy,
            xfl_strategy=xfl_strategy,
            xfl_param=xfl_param
        )
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        
        # Metrics collector
        self.metrics_collector = ServerMetricsCollector(db_path)
        
        # State management
        self.current_round = 0
        self.round_in_progress = False
        self.client_submissions = {}  # client_id -> submission dict
        self.lock = threading.Lock()
        
        # Test metrics
        self.test_loader = None
        
        print(f"✅ FLServer initialized")
        print(f"   Strategy: {self.aggregation_strategy.name}")
        print(f"   XFL: {xfl_strategy}")
        print(f"   Total rounds: {self.num_rounds}")
        print(f"   Clients per round: {self.clients_per_round}")
    
    def start_round(self) -> Dict[str, Any]:
        """Start a new FL round"""
        with self.lock:
            if self.current_round >= self.num_rounds:
                return {
                    "status": "completed",
                    "message": "All rounds completed"
                }
            
            if self.round_in_progress:
                return {
                    "status": "in_progress",
                    "round": self.current_round,
                    "message": "Round already in progress"
                }
            
            self.current_round += 1
            self.round_in_progress = True
            self.client_submissions = []  # Reset submissions list
            
            # Get XFL info for logging
            xfl_info = self.aggregation_strategy.get_xfl_info()
            print(f"\n🔄 Starting Round {self.current_round}/{self.num_rounds}")
            print(f"   XFL Strategy: {xfl_info['strategy']}")
            
            return {
                "status": "started",
                "round": self.current_round,
                "clients_expected": self.clients_per_round,
                "xfl_strategy": xfl_info['strategy']
            }
    
    def submit_client_update(
        self,
        client_id: int,
        model_weights: OrderedDict,
        num_samples: int,
        client_metrics: Dict[str, Any],
        quantization_meta: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """Receive model update from a client"""
        with self.lock:
            if not self.round_in_progress:
                return {
                    "status": "error",
                    "message": "No round in progress"
                }
            
            # Store submission
            self.client_submissions.append({
                "client_id": client_id,
                "weights": model_weights,
                "num_samples": num_samples,
                "metrics": client_metrics,
                "quantization_meta": quantization_meta
            })
            
            # Store client metrics in database
            self.metrics_collector.store_client_metrics(
                self.current_round,
                client_id,
                client_metrics
            )
            
            print(f"   ✅ Received update from Client {client_id} "
                  f"({len(self.client_submissions)}/{self.clients_per_round})")
            
            # Check if all clients submitted
            if len(self.client_submissions) >= self.clients_per_round:
                self._aggregate_round()
            
            return {
                "status": "received",
                "round": self.current_round,
                "submissions": len(self.client_submissions)
            }
    
    def _aggregate_round(self):
        """Aggregate client updates using XFL strategy"""
        xfl_info = self.aggregation_strategy.get_xfl_info()
        print(f"\n📊 Aggregating {len(self.client_submissions)} clients with XFL: {xfl_info['strategy']}")

        start_time = time.time()

        try:
            # Extract weights and sample counts
            client_weights = [sub["weights"] for sub in self.client_submissions]
            client_num_samples = [sub["num_samples"] for sub in self.client_submissions]

            # Check if any quantization metadata is present (for XFL quantization)
            has_quantization = any(sub.get("quantization_meta") for sub in self.client_submissions)

            if has_quantization and xfl_info['variant'] == 'quantization':
                print("   🔧 Applying dequantization before aggregation...")
                # Dequantize weights before aggregation
                client_weights = self._dequantize_client_weights(client_weights)

            # Aggregate with XFL
            aggregated_weights = self.aggregation_strategy.aggregate(
                client_weights,
                client_num_samples
            )

            # Update global model
            if isinstance(self.aggregation_strategy, XFL):
                # For XFL, aggregated_weights contains only updated layers
                current_state = self.global_model.state_dict()
                current_state.update(aggregated_weights)
                self.global_model.load_state_dict(current_state)
            else:
                # For FedAvg variants, aggregated_weights contains all layers
                self.global_model.load_state_dict(aggregated_weights)

            aggregation_time = time.time() - start_time

            # Evaluate global model
            test_loss, test_accuracy = None, None
            if self.test_loader is not None:
                try:
                    test_loss, test_accuracy = self._evaluate_global_model()
                except Exception as e:
                    print(f"⚠️  Global model evaluation failed: {e}")
                    test_loss, test_accuracy = None, None

            # Store round metrics
            self.metrics_collector.store_round_metrics(
                round_number=self.current_round,
                num_clients=len(self.client_submissions),
                aggregation_time=aggregation_time,
                global_test_loss=test_loss,
                global_test_accuracy=test_accuracy,
                total_samples=sum(client_num_samples)
            )

            print(f"✅ Round {self.current_round} completed in {aggregation_time:.2f}s")
            if test_accuracy is not None:
                print(f"   Global Test Accuracy: {test_accuracy:.2f}%")

        except Exception as e:
            aggregation_time = time.time() - start_time
            print(f"❌ Aggregation failed for round {self.current_round}: {e}")
            print(f"   Round will be completed anyway to prevent blocking")

        # Reset round state
        self.round_in_progress = False
    
    def _evaluate_global_model(self):
        """Evaluate global model on test set"""
        self.global_model.eval()
        device = next(self.global_model.parameters()).device
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def _dequantize_client_weights(self, client_weights: List[OrderedDict]) -> List[OrderedDict]:
        """
        Dequantize client weights before aggregation for XFL quantization

        Args:
            client_weights: List of quantized client weights (integers)

        Returns:
            List of dequantized client weights (floats)
        """
        dequantized_weights = []

        for i, client_weight in enumerate(client_weights):
            # Find the corresponding submission with quantization metadata
            client_meta = self.client_submissions[i].get("quantization_meta")

            if client_meta is None:
                # No quantization metadata, assume already dequantized
                dequantized_weights.append(client_weight)
                continue

            # Dequantize each parameter
            dequantized = OrderedDict()
            for param_name, param in client_weight.items():
                if param_name in client_meta and client_meta[param_name]['quantized']:
                    # This parameter was quantized, dequantize it
                    meta = client_meta[param_name]
                    # Convert integers back to float using scale
                    dequantized_param = param.float() * meta['scale']
                    dequantized[param_name] = dequantized_param
                else:
                    # Not quantized, keep as is
                    dequantized[param_name] = param.float()

            dequantized_weights.append(dequantized)

        return dequantized_weights

    def get_global_model(self) -> OrderedDict:
        """Get current global model weights"""
        with self.lock:
            return self.global_model.state_dict()
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status with XFL info"""
        with self.lock:
            xfl_info = self.aggregation_strategy.get_xfl_info()
            # Count logical layers (unique prefixes) instead of total parameters
            layer_names = list(self.global_model.state_dict().keys())
            num_layers = len(set(name.split('.')[0] for name in layer_names))
            return {
                "current_round": self.current_round,
                "total_rounds": self.num_rounds,
                "round_in_progress": self.round_in_progress,
                "submissions_received": len(self.client_submissions),
                "clients_expected": self.clients_per_round,
                "xfl_strategy": xfl_info['strategy'],
                "xfl_param": xfl_info['param'],
                "num_layers": num_layers
            }
    
    def set_xfl_strategy(self, strategy: str, param: int = 3) -> Dict[str, Any]:
        """
        Change XFL strategy dynamically

        Args:
            strategy: New XFL strategy
            param: New XFL parameter

        Returns:
            Status dictionary
        """
        with self.lock:
            if self.round_in_progress:
                return {
                    "status": "error",
                    "message": "Cannot change strategy during active round"
                }

            if strategy.startswith("xfl"):
                # For XFL strategies, create new XFL aggregation strategy
                self.aggregation_strategy = create_aggregation_strategy("xfl", strategy, param)
            else:
                # For non-XFL strategies (like all_layers), create new FedAvg strategy
                self.aggregation_strategy = FedAvg(xfl_strategy=strategy, xfl_param=param)

            return {
                "status": "success",
                "message": f"XFL strategy updated to {strategy}",
                "xfl_strategy": strategy,
                "xfl_param": param
            }


# Flask app
app = Flask(__name__)
CORS(app)

# Global server instance
fl_server: FLServer = None


def weights_to_base64(weights: OrderedDict) -> str:
    """Serialize weights to base64 string"""
    weights_bytes = pickle.dumps(weights)
    return base64.b64encode(weights_bytes).decode('utf-8')


def base64_to_weights(base64_str: str) -> OrderedDict:
    """Deserialize weights from base64 string"""
    weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
    return pickle.loads(weights_bytes)


@app.route('/status', methods=['GET'])
def status():
    """Get server status with XFL info"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    return jsonify(fl_server.get_server_status())


@app.route('/start_round', methods=['POST'])
def start_round():
    """Start a new FL round"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    result = fl_server.start_round()
    return jsonify(result)


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Get current global model weights"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    weights = fl_server.get_global_model()
    weights_b64 = weights_to_base64(weights)

    # Include XFL info for client to decide on partial updates
    xfl_info = fl_server.aggregation_strategy.get_xfl_info()

    return jsonify({
        "weights": weights_b64,
        "round": fl_server.current_round,
        "xfl_strategy": xfl_info['strategy'],
        "xfl_param": xfl_info['param'],
        "sparsification_threshold": xfl_info.get('sparsification_threshold', 0.01),
        "quantization_bits": xfl_info.get('quantization_bits', 8)
    })


@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receive client model update"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.get_json()

    # Extract data
    client_id = data.get('client_id')
    weights_b64 = data.get('weights')
    num_samples = data.get('num_samples')
    client_metrics = data.get('metrics', {})
    quantization_meta = data.get('quantization_meta')

    # Validate
    if client_id is None or weights_b64 is None or num_samples is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Deserialize weights
    weights = base64_to_weights(weights_b64)

    # Submit to server
    result = fl_server.submit_client_update(
        client_id=client_id,
        model_weights=weights,
        num_samples=num_samples,
        client_metrics=client_metrics,
        quantization_meta=quantization_meta
    )
    
    return jsonify(result)


@app.route('/xfl/set_strategy', methods=['POST'])
def set_xfl_strategy():
    """
    Set XFL strategy dynamically
    
    POST body:
    {
        "strategy": "first_n_layers",
        "param": 2
    }
    """
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.get_json()
    strategy = data.get('strategy', 'all_layers')
    param = data.get('param', 3)
    
    result = fl_server.set_xfl_strategy(strategy, param)
    return jsonify(result)


@app.route('/xfl/get_strategy', methods=['GET'])
def get_xfl_strategy():
    """Get current XFL strategy"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    xfl_info = fl_server.aggregation_strategy.get_xfl_info()
    return jsonify(xfl_info)


@app.route('/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get metrics summary"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    summary = fl_server.metrics_collector.get_summary_statistics()
    return jsonify(summary)


def create_server(
    global_model: torch.nn.Module,
    test_loader = None,
    aggregation_strategy: str = "fedavg",
    num_rounds: int = 10,
    clients_per_round: int = 5,
    db_path: str = "logs/server_metrics.db",
    xfl_strategy: str = "all_layers",
    xfl_param: int = 3
) -> FLServer:
    """
    Create and initialize FL server with XFL support
    """
    global fl_server
    
    fl_server = FLServer(
        global_model=global_model,
        aggregation_strategy=aggregation_strategy,
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        db_path=db_path,
        xfl_strategy=xfl_strategy,
        xfl_param=xfl_param
    )
    
    fl_server.test_loader = test_loader
    
    return fl_server


def run_server(host: str = "localhost", port: int = 5000, debug: bool = False):
    """Run Flask server"""
    print(f"\n🚀 Starting FL Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)