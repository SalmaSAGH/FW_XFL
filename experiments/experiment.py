"""
Main experiment orchestration script for XFL-RPiLab
Coordinates FL server and multiple clients
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
import requests
from typing import List
import torch

from config import load_config
from client import create_model, create_dataloaders, FLClient
from server import create_server, run_server


class ExperimentOrchestrator:
    """
    Orchestrates FL experiments with server and multiple clients
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        print("="*70)
        print("XFL-RPiLab Experiment Orchestrator")
        print("="*70)
        
        # Load configuration
        print("\n📋 Loading configuration...")
        self.config = load_config(config_path)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.experiment.seed)
        
        # Initialize components
        self.server = None
        self.clients = []
        self.server_thread = None
        
        print(f"\n✅ Orchestrator initialized")
        print(f"   Experiment: {self.config.experiment.name}")
        print(f"   Description: {self.config.experiment.description}")
    
    def setup_model_and_data(self):
        """Setup model and data loaders"""
        print("\n" + "="*70)
        print("Setting Up Model and Data")
        print("="*70)
        
        # Create model
        print(f"\n📊 Creating model: {self.config.model.name}")
        self.global_model = create_model(
            model_name=self.config.model.name,
            num_classes=self.config.model.num_classes
        )
        
        # Create data loaders
        print(f"\n📊 Creating data loaders for {self.config.dataset.name}...")
        self.client_loaders, self.test_loader = create_dataloaders(
            dataset_name=self.config.dataset.name,
            num_clients=self.config.federated_learning.num_clients,
            batch_size=self.config.dataset.batch_size,
            distribution=self.config.dataset.data_distribution,
            data_dir=self.config.paths.data_dir,
            seed=self.config.experiment.seed
        )
        
        print(f"\n✅ Model and data setup completed")
    
    def start_server(self):
        """Start FL server in a separate thread"""
        print("\n" + "="*70)
        print("Starting FL Server")
        print("="*70)
        
        # Create server
        self.server = create_server(
            global_model=self.global_model,
            test_loader=self.test_loader,
            aggregation_strategy=self.config.server.aggregation_method,
            num_rounds=self.config.federated_learning.num_rounds,
            clients_per_round=self.config.federated_learning.clients_per_round,
            db_url=self.config.server.metrics_db_url
        )
        
        # Start server in separate thread
        self.server_thread = threading.Thread(
            target=run_server,
            kwargs={
                "host": self.config.server.host,
                "port": self.config.server.port,
                "debug": False
            },
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready
        print(f"\n⏳ Waiting for server to start...")
        server_url = f"http://{self.config.server.host}:{self.config.server.port}"
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{server_url}/status", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Server is ready at {server_url}")
                    time.sleep(1)  # Extra wait for stability
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError("Failed to start server")
    
    def create_clients(self):
        """Create FL clients"""
        print("\n" + "="*70)
        print("Creating FL Clients")
        print("="*70)
        
        server_url = f"http://{self.config.server.host}:{self.config.server.port}"
        
        for client_id in range(self.config.federated_learning.clients_per_round):
            # Create a copy of the model for this client
            client_model = create_model(
                model_name=self.config.model.name,
                num_classes=self.config.model.num_classes
            )
            
            # Create FL client
            fl_client = FLClient(
                client_id=client_id,
                model=client_model,
                train_loader=self.client_loaders[client_id],
                server_url=server_url,
                optimizer=self.config.training.optimizer,
                learning_rate=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
                local_epochs=self.config.federated_learning.local_epochs,
                timeout=self.config.client.timeout_sec
            )
            
            self.clients.append(fl_client)
            print(f"   ✅ Client {client_id} created")
        
        print(f"\n✅ {len(self.clients)} clients created")
    
    def run_round(self, round_num: int):
        """
        Run one FL round
        
        Args:
            round_num: Current round number
        """
        print("\n" + "="*70)
        print(f"Round {round_num}/{self.config.federated_learning.num_rounds}")
        print("="*70)
        
        # Start round on server
        server_url = f"http://{self.config.server.host}:{self.config.server.port}"
        print(f"\n📤 Starting round {round_num} on server...")
        
        try:
            response = requests.post(f"{server_url}/start_round", timeout=10)
            result = response.json()
            print(f"   Server status: {result.get('status')}")
        except Exception as e:
            print(f"   ❌ Error starting round: {e}")
            return False
        
        # Run clients in parallel
        print(f"\n🚀 Running {len(self.clients)} clients in parallel...")
        
        client_threads = []
        for client in self.clients:
            thread = threading.Thread(
                target=client.participate_in_round,
                kwargs={"verbose": False},
                daemon=True
            )
            thread.start()
            client_threads.append(thread)
        
        # Wait for all clients to finish
        for i, thread in enumerate(client_threads):
            thread.join()
            print(f"   ✅ Client {i} completed")
        
        # Wait a bit for server to aggregate
        time.sleep(2)
        
        # Get server status
        try:
            response = requests.get(f"{server_url}/status", timeout=10)
            status = response.json()
            print(f"\n📊 Round {round_num} Summary:")
            print(f"   Submissions received: {status.get('submissions_received', 0)}")
            print(f"   Round in progress: {status.get('round_in_progress', False)}")
        except Exception as e:
            print(f"   ⚠️  Could not get server status: {e}")
        
        return True
    
    def run_experiment(self):
        """Run complete FL experiment"""
        try:
            # Setup
            self.setup_model_and_data()
            
            # Start server
            self.start_server()
            
            # Create clients
            self.create_clients()
            
            # Run FL rounds
            print("\n" + "="*70)
            print("Starting Federated Learning Rounds")
            print("="*70)
            
            experiment_start_time = time.time()
            
            for round_num in range(1, self.config.federated_learning.num_rounds + 1):
                round_start = time.time()
                
                success = self.run_round(round_num)
                
                round_time = time.time() - round_start
                print(f"\n⏱️  Round {round_num} completed in {round_time:.2f}s")
                
                if not success:
                    print(f"❌ Round {round_num} failed")
                    break
                
                # Short pause between rounds
                if round_num < self.config.federated_learning.num_rounds:
                    time.sleep(2)
            
            total_time = time.time() - experiment_start_time
            
            # Final summary
            print("\n" + "="*70)
            print("Experiment Completed!")
            print("="*70)
            print(f"\n📊 Final Summary:")
            print(f"   Total rounds: {self.config.federated_learning.num_rounds}")
            print(f"   Total clients: {len(self.clients)}")
            print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"   Average time per round: {total_time/self.config.federated_learning.num_rounds:.2f}s")
            
            # Get final metrics
            server_url = f"http://{self.config.server.host}:{self.config.server.port}"
            try:
                response = requests.get(f"{server_url}/metrics/summary", timeout=10)
                summary = response.json()
                
                print(f"\n📈 Performance Metrics:")
                if summary.get('rounds'):
                    rounds_data = summary['rounds']
                    print(f"   Average Accuracy: {rounds_data.get('avg_accuracy', 'N/A')}%")
                    print(f"   Max Accuracy: {rounds_data.get('max_accuracy', 'N/A')}%")
                    print(f"   Avg Aggregation Time: {rounds_data.get('avg_aggregation_time', 'N/A')}s")
                
                if summary.get('clients'):
                    clients_data = summary['clients']
                    print(f"\n💻 Client Metrics:")
                    print(f"   Avg Training Time: {clients_data.get('avg_training_time', 'N/A')}s")
                    print(f"   Avg Client Accuracy: {clients_data.get('avg_accuracy', 'N/A')}%")
                    print(f"   Avg CPU Usage: {clients_data.get('avg_cpu_usage', 'N/A')}%")
                    print(f"   Avg Memory Usage: {clients_data.get('avg_memory_usage', 'N/A')} MB")
            
            except Exception as e:
                print(f"\n⚠️  Could not retrieve final metrics: {e}")
            
            print(f"\n✅ Experiment '{self.config.experiment.name}' completed successfully!")
            print(f"📁 Metrics saved to PostgreSQL database: {self.config.server.metrics_db_url}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Experiment interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Experiment failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "="*70)
            print("Shutting Down")
            print("="*70)
            print("Server will be stopped when script exits...")


def main():
    """Main entry point"""
    # Check for config file argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/config.yaml"
    
    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(config_path)
    orchestrator.run_experiment()


if __name__ == "__main__":
    main()