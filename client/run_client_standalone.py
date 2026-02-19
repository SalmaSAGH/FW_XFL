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
            if requests.get(f"{server_url}/api/status", timeout=2).status_code == 200:
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
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        distribution=args.distribution,
        data_dir="/app/data",
        seed=42
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
    
    # Main loop - AGGRESSIVE polling with exponential backoff retry
    last_participated = 0
    check_count = 0
    consecutive_failures = 0
    max_retries = 5
    base_delay = 0.5  # Start with 0.5 second delay
    
    while True:
        try:
            check_count += 1
            
            # Get status
            resp = requests.get(f"{server_url}/api/status", timeout=2)
            status = resp.json()
            
            current_round = status.get('current_round', 0)
            
            # Log every 10 checks (for debugging)
            if check_count % 10 == 0:
                print(f"Client {args.client_id}: Monitoring... (Round: {current_round}, Last: {last_participated})")
            
            # NEW ROUND DETECTED?
            if current_round > last_participated and current_round > 0:
                print(f"\n🔥 Client {args.client_id}: Round {current_round} DETECTED! Participating NOW...")
                
                # Retry logic with exponential backoff
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    start_time = time.time()
                    success = fl_client.participate_in_round(verbose=False)
                    elapsed = time.time() - start_time
                    
                    if success:
                        last_participated = current_round
                        consecutive_failures = 0
                        print(f"✅ Client {args.client_id}: Round {current_round} DONE in {elapsed:.1f}s\n")
                        break
                    else:
                        retry_count += 1
                        consecutive_failures += 1
                        
                        if retry_count < max_retries:
                            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s...
                            delay = base_delay * (2 ** (retry_count - 1))
                            print(f"⚠️  Client {args.client_id}: Attempt {retry_count}/{max_retries} failed. Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            print(f"❌ Client {args.client_id}: All {max_retries} attempts failed for round {current_round}")
            
            # Reset failure count on successful round
            if current_round == last_participated:
                consecutive_failures = 0
            
            # Adaptive sleep - check more frequently when server might be busy
            # But back off if there are consecutive failures
            if consecutive_failures > 3:
                time.sleep(2.0)  # Back off to 2s if having issues
            else:
                # Very short sleep - check every 200ms
                time.sleep(0.2)
        
        except KeyboardInterrupt:
            print(f"Client {args.client_id} stopped by user")
            break
        except Exception as e:
            # On exception, use exponential backoff
            delay = base_delay * (2 ** min(consecutive_failures, 5))
            print(f"⚠️  Client {args.client_id}: Connection error - {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


if __name__ == "__main__":
    main()