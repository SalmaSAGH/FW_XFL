"""
Standalone client for Raspberry Pi - Real Hardware Mode
Collects real metrics from Raspberry Pi hardware
"""

import sys
import os
import time
import requests
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FLClient, create_model, create_dataloaders
from client.model import DATASET_CONFIG
from client.dataset import create_single_client_loader
from client.real_hardware_metrics import RealHardwareMetricsCollector


def wait_for_server(server_url: str, timeout: int = 60):
    """Wait for server to be ready"""
    print(f"Waiting for server at {server_url}...")
    for i in range(timeout):
        try:
            response = requests.get(f"{server_url}/api/status", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"⚠️ Error checking server: {e}")
        time.sleep(1)
    raise RuntimeError(f"Server not reachable after {timeout}s")


def register_with_server(server_url: str, client_id: int, ip_address: str = None):
    """Register this client with the FL server"""
    import socket
    if ip_address is None:
        # Get the actual network IP by connecting to an external address
        try:
            # Connect to a public DNS server to determine the route
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            # This doesn't actually send data, just determines the route
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception:
            # Fallback: get IP from hostname resolution
            ip_address = socket.gethostbyname(socket.gethostname())
    
    hostname = socket.gethostname()
    username = "pi1"  # Default username
    
    print(f"Registering client {client_id} ({ip_address}) with server...")
    
    try:
        response = requests.post(
            f"{server_url}/api/physical/register",
            json={
                "client_id": client_id,
                "ip_address": ip_address,
                "hostname": hostname,
                "username": username
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Client {client_id} registered successfully!")
            print(f"   Server response: {response.json()}")
            return True
        else:
            print(f"⚠️ Registration failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False


def fetch_server_config(server_url: str) -> dict:
    """Fetch the latest FL configuration from the server."""
    try:
        response = requests.get(f"{server_url}/api/config", timeout=10)
        if response.status_code == 200:
            return response.json().get('config', {})
    except Exception:
        pass
    return {}


def apply_server_config_to_client(
    config: dict,
    fl_client,
    current_model: str,
    current_dataset: str,
    current_distribution: str,
    current_num_clients: int,
    current_batch_size: int,
    current_local_epochs: int,
    current_learning_rate: float
):
    """Apply server config changes to the running FL client."""
    if not config:
        return (
            current_model,
            current_dataset,
            current_distribution,
            current_num_clients,
            current_batch_size,
            current_local_epochs,
            current_learning_rate
        )

    server_model = config.get('model', current_model)
    server_dataset = config.get('dataset', current_dataset)
    server_distribution = config.get('dataDistribution', current_distribution)
    server_num_clients = int(config.get('numClients', current_num_clients))
    server_batch_size = int(config.get('batchSize', current_batch_size))
    server_local_epochs = int(config.get('localEpochs', current_local_epochs))
    server_learning_rate = float(config.get('learningRate', current_learning_rate))

    if server_local_epochs != fl_client.local_epochs:
        print(f"Client {fl_client.client_id}: 🔄 local_epochs updated {fl_client.local_epochs} → {server_local_epochs}")
        fl_client.local_epochs = server_local_epochs

    if server_learning_rate != fl_client.learning_rate:
        print(f"Client {fl_client.client_id}: 🔄 learning_rate updated {fl_client.learning_rate} → {server_learning_rate}")
        fl_client.learning_rate = server_learning_rate
        fl_client.trainer.learning_rate = server_learning_rate
        for param_group in fl_client.trainer.optimizer.param_groups:
            param_group['lr'] = server_learning_rate

    fl_client.reload_data_if_needed(
        dataset_name=server_dataset,
        model_name=server_model,
        distribution=server_distribution,
        num_clients=server_num_clients,
        batch_size=server_batch_size,
        verbose=False
    )

    return (
        server_model,
        server_dataset,
        server_distribution,
        server_num_clients,
        server_batch_size,
        server_local_epochs,
        server_learning_rate
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id',    type=int, required=True)
    parser.add_argument('--server-host',  type=str, default='localhost')
    parser.add_argument('--server-port',  type=int, default=5000)
    parser.add_argument('--server-url',   type=str, default=None,
                        help='Full server URL (overrides host/port)')
    parser.add_argument('--dataset',      type=str, default='MNIST')
    parser.add_argument('--num-clients',  type=int, default=40)
    parser.add_argument('--batch-size',   type=int, default=32)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--distribution', type=str, default='iid')
    parser.add_argument('--data-dir',     type=str, default='/home/pi1/XFL-RPiLab/data')
    parser.add_argument('--mode',         type=str, default='simulated',
                        choices=['simulated', 'real-hardware'],
                        help='Metrics collection mode')
    
    args = parser.parse_args()
    
    # Determine server URL
    if args.server_url:
        server_url = args.server_url
    else:
        server_url = f"http://{args.server_host}:{args.server_port}"
    
    print(f"\n{'='*60}")
    print(f"Raspberry Pi FL Client")
    print(f"{'='*60}")
    print(f"Client ID:     {args.client_id}")
    print(f"Server URL:    {server_url}")
    print(f"Dataset:       {args.dataset}")
    print(f"Mode:          {args.mode}")
    print(f"{'='*60}\n")
    
    # Wait for server
    wait_for_server(server_url)
    
    # Get the network IP address
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except Exception:
        ip_address = socket.gethostbyname(socket.gethostname())
    
    # Register with server as physical client
    register_with_server(server_url, args.client_id, ip_address)
    
    # ── Get current configuration from server ─────────────────────────────────
    print(f"Client {args.client_id}: Getting configuration from server...")
    config = fetch_server_config(server_url)
    server_model = config.get('model', 'SimpleCNN')
    server_dataset = config.get('dataset', args.dataset)
    server_num_clients = int(config.get('numClients', args.num_clients))
    server_batch_size = int(config.get('batchSize', args.batch_size))
    server_local_epochs = int(config.get('localEpochs', args.local_epochs))
    server_learning_rate = float(config.get('learningRate', 0.01))
    server_distribution = config.get('dataDistribution', args.distribution)

    print(f"Client {args.client_id}: Server config - model={server_model}, dataset={server_dataset}, "
          f"num_clients={server_num_clients}, batch_size={server_batch_size}, local_epochs={server_local_epochs}")

    # ── Create model using server configuration ──────────────────────────────
    # Get parameters for the server-selected model
    from client.model import get_model_params_for_dataset
    num_classes, in_channels, input_size = get_model_params_for_dataset(server_model, server_dataset)
    
    print(f"Creating model: {server_model} ({num_classes} classes, {in_channels}ch, {input_size}px)")
    model = create_model(
        server_model,
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size
    )
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_loader = create_single_client_loader(
        dataset_name=server_dataset,
        client_id=args.client_id,
        num_clients=server_num_clients,
        batch_size=server_batch_size,
        distribution=server_distribution,
        data_dir=args.data_dir,
        seed=42
    )
    print(f"Data loaded: {len(train_loader.dataset)} samples")
    
    # Create FL client
    fl_client = FLClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        server_url=server_url,
        local_epochs=server_local_epochs,
        timeout=120,
        dataset_name=server_dataset,        
        num_clients=server_num_clients,     
        batch_size=server_batch_size,       
        distribution=server_distribution,  
        data_dir=args.data_dir,
        use_real_hardware_metrics=(args.mode == 'real-hardware'),
        real_metrics_collector=None  # Will be set after initialization
    )
    
    # Initialize real hardware metrics if in real-hardware mode
    metrics_collector = None
    if args.mode == 'real-hardware':
        print("\n📊 Initializing Real Hardware Metrics Collector...")
        metrics_collector = RealHardwareMetricsCollector(
            client_id=args.client_id,
            server_url=server_url,  # Pass server_url for ping measurements
            collection_interval=1.0
        )
        metrics_collector.start_collection()
        # Connect real metrics collector to FLClient
        fl_client.real_metrics_collector = metrics_collector
        print("✅ Real hardware metrics collection started")
    
    print(f"\n✅ Client {args.client_id} READY and waiting for rounds\n")
    
    # Main polling loop
    round_num = 0
    last_round = 0
    server_was_available = True  # Track server availability
    
    # Initial registration
    print(f"🔄 Ensuring client is registered...")
    register_with_server(server_url, args.client_id, ip_address)
    
    # Make ip_address available for re-registration
    client_ip_address = ip_address
    
    last_heartbeat = 0
    heartbeat_interval = 20

    def send_heartbeat():
        try:
            requests.post(
                f"{server_url}/api/physical/heartbeat",
                json={
                    "client_id": args.client_id,
                    "status": "training" if round_in_progress else "connected",
                    "round_number": round_num
                },
                timeout=5
            )
        except Exception:
            pass

    while True:
        try:
            current_time = time.time()
            
            # Check server status for active round
            response = requests.get(
                f"{server_url}/api/status",
                timeout=10
            )
            
            # Server is available
            if not server_was_available:
                print(f"✅ Server is back online, re-registering...")
                register_with_server(server_url, args.client_id, client_ip_address)
                server_was_available = True
            
            if response.status_code == 200:
                status = response.json()

                # Reload latest config from server before participating
                server_config = fetch_server_config(server_url)
                (server_model,
                 server_dataset,
                 server_distribution,
                 server_num_clients,
                 server_batch_size,
                 server_local_epochs,
                 server_learning_rate) = apply_server_config_to_client(
                     server_config,
                     fl_client,
                     server_model,
                     server_dataset,
                     server_distribution,
                     server_num_clients,
                     server_batch_size,
                     server_local_epochs,
                     server_learning_rate
                 )
                
                # Check if round is in progress and this client is selected
                round_in_progress = status.get('round_in_progress', False)
                selected_clients = status.get('selected_clients', [])
                current_round = status.get('current_round', 0)
                
                if current_time - last_heartbeat >= heartbeat_interval:
                    heartbeat_payload = {
                        "client_id": args.client_id,
                        "status": "training" if round_in_progress and args.client_id in selected_clients else "connected",
                        "round_number": current_round
                    }
                    try:
                        requests.post(
                            f"{server_url}/api/physical/heartbeat",
                            json=heartbeat_payload,
                            timeout=5
                        )
                    except Exception:
                        pass
                    last_heartbeat = current_time

                if round_in_progress and args.client_id in selected_clients and current_round > last_round:
                    last_round = current_round
                    round_num = current_round
                    print(f"\n🔄 Round {round_num} started - Participating...")
                    
                    # Reset metrics for the new round (if using real hardware mode)
                    if metrics_collector is not None:
                        metrics_collector.reset_round_metrics()
                    
                    # Participate in round (already sends weights + metrics to server)
                    result = fl_client.participate_in_round()
                    
                    print(f"✅ Round {round_num} completed")
            
            time.sleep(5)  # Poll every 5 seconds
            
        except requests.exceptions.ConnectionError:
            # Server not available
            if server_was_available:
                print(f"⚠️ Server unavailable, will retry...")
                server_was_available = False
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down client...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down client...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(5)
    
    # Cleanup
    if metrics_collector:
        metrics_collector.stop_collection()
        print("\n📊 Final Metrics Summary:")
        summary = metrics_collector.get_summary()
        print(f"  CPU Avg: {summary['cpu']['avg_percent']}%")
        print(f"  Memory Avg: {summary['memory']['avg_percent']}%")
        if summary['temperature']['avg_celsius']:
            print(f"  Temperature Avg: {summary['temperature']['avg_celsius']}°C")


if __name__ == "__main__":
    main()