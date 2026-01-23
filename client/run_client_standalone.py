"""
ULTRA-FAST Standalone Client
Optimized for minimal latency and maximum throughput
"""

import sys
import os
import time
import requests
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FLClient, create_model, create_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', type=int, required=True)
    parser.add_argument('--server-host', type=str, default='server')
    parser.add_argument('--server-port', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--distribution', type=str, default='iid')
    
    args = parser.parse_args()
    
    server_url = f"http://{args.server_host}:{args.server_port}"
    
    # Wait for server (quick)
    print(f"[{args.client_id}] Waiting for server...")
    for _ in range(20):
        try:
            if requests.get(f"{server_url}/status", timeout=1).status_code == 200:
                break
        except:
            pass
        time.sleep(0.5)
    
    print(f"[{args.client_id}] Loading data...")
    
    # Create model
    model = create_model("SimpleCNN", num_classes=10)
    
    # Load dataset (cache it in memory)
    client_loaders, _ = create_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        distribution=args.distribution,
        data_dir="/app/data",
        seed=42
    )
    train_loader = client_loaders[args.client_id]
    
    print(f"[{args.client_id}] Creating FL client...")
    
    # Create FL client
    fl_client = FLClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        server_url=server_url,
        local_epochs=args.local_epochs,
        timeout=60
    )
    
    print(f"✅ [{args.client_id}] READY ({len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches)\n")
    
    # Main loop - CHECK EVERY 100ms
    last_round = 0
    
    while True:
        try:
            # Quick status check
            resp = requests.get(f"{server_url}/status", timeout=1)
            status = resp.json()
            
            current_round = status.get('current_round', 0)
            
            # New round?
            if current_round > last_round and current_round > 0:
                print(f"\n🔥 [{args.client_id}] Round {current_round} START")
                
                t0 = time.time()
                success = fl_client.participate_in_round(verbose=True)
                elapsed = time.time() - t0
                
                if success:
                    last_round = current_round
                    print(f"✅ [{args.client_id}] Round {current_round} DONE in {elapsed:.1f}s\n")
                else:
                    print(f"❌ [{args.client_id}] Round {current_round} FAILED\n")
            
            # Sleep briefly
            time.sleep(0.1)  # Check 10 times per second
        
        except KeyboardInterrupt:
            break
        except:
            time.sleep(0.2)


if __name__ == "__main__":
    main()