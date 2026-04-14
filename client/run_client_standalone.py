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
from client.model import DATASET_CONFIG
from client.dataset import create_single_client_loader

def wait_for_server(server_url: str):
    """Wait for server to be ready"""
    for i in range(30):
        try:
            if requests.get(f"{server_url}/api/status", timeout=5).status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    raise RuntimeError("Server not reachable after 30s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id',    type=int, required=True)
    parser.add_argument('--server-host',  type=str, default='server')
    parser.add_argument('--server-port',  type=int, default=5000)
    parser.add_argument('--dataset',      type=str, default='MNIST')
    parser.add_argument('--num-clients',  type=int, default=40)
    parser.add_argument('--batch-size',   type=int, default=32)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--distribution', type=str, default='iid')

    args = parser.parse_args()

    print(f"Client {args.client_id} starting...")

    server_url = f"http://{args.server_host}:{args.server_port}"

    # ── Wait for server ───────────────────────────────────────────────────────
    wait_for_server(server_url)
    print(f"Client {args.client_id}: Server ready at {server_url}")

    # ── Create model using DATASET_CONFIG ─────────────────────────────────────
    dataset_cfg = DATASET_CONFIG.get(args.dataset, ('SimpleCNN', 10, 1, 28))
    model_name, num_classes, in_channels, input_size = dataset_cfg

    print(f"Client {args.client_id}: Creating model={model_name} | "
          f"num_classes={num_classes} | in_channels={in_channels} | input_size={input_size}")

    model = create_model(
        model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size
    )

    # ── Load initial data ─────────────────────────────────────────────────────
    print(f"Client {args.client_id}: Loading data ({args.dataset}, {args.distribution})...")
    start_load = time.time()

    train_loader = create_single_client_loader(
        dataset_name=args.dataset,
        client_id=args.client_id,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        distribution=args.distribution,
        data_dir="/app/data",
        seed=42
        )

    print(f"Client {args.client_id}: Data loaded in {time.time() - start_load:.1f}s "
          f"({len(train_loader.dataset)} samples)")

    # ── Create FL client — passer TOUS les paramètres ────────────────────────
    fl_client = FLClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        server_url=server_url,
        local_epochs=args.local_epochs,
        timeout=120,
        dataset_name=args.dataset,        
        num_clients=args.num_clients,     
        batch_size=args.batch_size,       
        distribution=args.distribution,  
        data_dir="/app/data"             
    )

    print(f"✅ Client {args.client_id} READY and waiting for rounds\n")

    # ── Main polling loop ─────────────────────────────────────────────────────
    last_participated = 0
    check_count = 0
    consecutive_failures = 0
    max_retries = 5
    base_delay = 1.0

    while True:
        try:
            check_count += 1

            resp = requests.get(f"{server_url}/api/status", timeout=5)
            status = resp.json()
            current_round = status.get('current_round', 0)

            if check_count % 10 == 0:
                print(f"Client {args.client_id}: Monitoring... "
                      f"(Round: {current_round}, Last: {last_participated})")

            if current_round > last_participated and current_round > 0:
                print(f"\n🔥 Client {args.client_id}: Round {current_round} DETECTED! Participating NOW...")

                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    start_time = time.time()
                    # verbose=True pour voir toutes les erreurs dans les logs Docker
                    success = fl_client.participate_in_round(verbose=True)
                    elapsed = time.time() - start_time

                    if success:
                        last_participated = current_round
                        consecutive_failures = 0
                        print(f"✅ Client {args.client_id}: Round {current_round} "
                              f"DONE in {elapsed:.1f}s\n")
                        break
                    else:
                        retry_count += 1
                        consecutive_failures += 1

                        if retry_count < max_retries:
                            delay = base_delay * (2 ** (retry_count - 1))
                            print(f"⚠️  Client {args.client_id}: Attempt {retry_count}/{max_retries} "
                                  f"failed. Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            print(f"❌ Client {args.client_id}: All {max_retries} attempts "
                                  f"failed for round {current_round}")

            if current_round == last_participated:
                consecutive_failures = 0

            if consecutive_failures > 3:
                time.sleep(2.0)
            else:
                time.sleep(1.0)

        except KeyboardInterrupt:
            print(f"Client {args.client_id} stopped by user")
            break
        except Exception as e:
            delay = base_delay * (2 ** min(consecutive_failures, 5))
            print(f"⚠️  Client {args.client_id}: Connection error — {e}. "
                  f"Retrying in {delay:.1f}s...")
            time.sleep(delay)


if __name__ == "__main__":
    main()