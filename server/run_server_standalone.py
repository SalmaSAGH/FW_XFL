"""
Standalone server script for Docker deployment
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import create_server
from client import create_model, create_dataloaders

# Import Flask app directly to configure it
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from server.server import app


def main():
    """Main entry point for standalone server"""
    
    # Read number of clients from environment variable
    num_clients = int(os.getenv('NUM_CLIENTS', '5'))
    
    print("="*70)
    print("XFL-RPiLab FL Server (Docker)")
    print("="*70)
    print(f"Starting server on 0.0.0.0:5000")
    print(f"Expected clients: {num_clients}")
    print("="*70 + "\n")
    
    # Create model
    print("📊 Creating global model...")
    model = create_model("SimpleCNN", num_classes=10)
    
    # Create test dataloader
    print("\n📊 Loading test dataset...")
    _, test_loader = create_dataloaders(
        dataset_name="MNIST",
        num_clients=5,  # Doesn't matter for test set
        batch_size=64,
        distribution="iid",
        data_dir="/app/data",
        seed=42
    )
    
    # Create server
    print("\n📊 Creating FL server...")
    server = create_server(
        global_model=model,
        test_loader=test_loader,
        aggregation_strategy="fedavg",
        num_rounds=100,  # High number, will be controlled via API
        clients_per_round=num_clients,  # From environment variable
        db_path="/app/logs/server_metrics.db"
    )
    
    # IMPORTANT: Do NOT start a round automatically
    # Rounds will be started manually via dashboard or API
    print("\n⚠️  Server is ready but NO round started.")
    print("   Use the dashboard or API to start rounds manually.")
    
    print("\n✅ Server initialization complete!")
    print("🚀 Starting Flask server...\n")
    
    # Run server with more threads for better concurrency
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, 
            processes=1, use_reloader=False)


if __name__ == "__main__":
    main()