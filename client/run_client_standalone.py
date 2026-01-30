"""
Standalone client - OPTIMIZED FOR SPEED
Participates immediately when round starts
"""

import sys
import os
import time
import requests
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FLClient, create_model, create_dataloaders


def wait_for_server(server_url: str):
    """Wait for server"""
    for i in range(30):
        try:
            if requests.get(f"{server_url}/status", timeout=2).status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    raise RuntimeError("Server not reachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', type=int, required=True)
    parser.add_argument('--server-host', type=str, default='server')
    parser.add_argument('--server-port', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--num-clients', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--distribution', type=str, default='iid')
    
    args = parser.parse_args()
    
    print(f"Client {args.client_id} starting...")
    
    server_url = f"http://{args.server_host}:{args.server_port}"
    
    # Wait for server
    wait_for_server(server_url)
    print(f"Client {args.client_id}: Server ready")
    
    # Create model
    model = create_model("SimpleCNN", num_classes=10)
    
    # Load data (this is the SLOW part - do it ONCE)
    print(f"Client {args.client_id}: Loading data...")
    start_load = time.time()
    client_loaders, _ = create_dataloaders(
        dataset_name=args.dataset,
        num_clients=1, 
        batch_size=args.batch_size,
        distribution=args.distribution,
        data_dir="/app/data",
        seed=42 + args.client_id
    )
    train_loader = client_loaders[args.client_id]
    print(f"Client {args.client_id}: Data loaded in {time.time()-start_load:.1f}s")
    
    # Create FL client
    fl_client = FLClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        server_url=server_url,
        local_epochs=args.local_epochs,
        timeout=120
    )
    
    print(f"✅ Client {args.client_id} READY and waiting for rounds\n")
    
    # Main loop - AGGRESSIVE polling
    last_participated = 0
    check_count = 0
    
    while True:
        try:
            check_count += 1
            
            # Get status
            resp = requests.get(f"{server_url}/status", timeout=2)
            status = resp.json()
            
            current_round = status.get('current_round', 0)
            
            # Log every 10 checks (for debugging)
            if check_count % 10 == 0:
                print(f"Client {args.client_id}: Monitoring... (Round: {current_round}, Last: {last_participated})")
            
            # NEW ROUND DETECTED?
            if current_round > last_participated and current_round > 0:
                print(f"\n🔥 Client {args.client_id}: Round {current_round} DETECTED! Participating NOW...")
                
                start_time = time.time()
                success = fl_client.participate_in_round(verbose=False)
                elapsed = time.time() - start_time
                
                if success:
                    last_participated = current_round
                    print(f"✅ Client {args.client_id}: Round {current_round} DONE in {elapsed:.1f}s\n")
                else:
                    print(f"⚠️  Client {args.client_id}: Failed, will retry\n")
            
            # Very short sleep - check every 200ms
            time.sleep(0.2)
        
        except KeyboardInterrupt:
            print(f"Client {args.client_id} stopped by user")
            break
        except:
            time.sleep(0.5)


if __name__ == "__main__":
    main()