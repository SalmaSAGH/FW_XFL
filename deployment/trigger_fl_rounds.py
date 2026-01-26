"""
Manual trigger for FL rounds - use this to manually start rounds
Usage: python deployment/trigger_fl_rounds.py --rounds 10
"""

import requests
import time
import argparse


def trigger_rounds(server_url: str, num_rounds: int, clients_per_round: int):
    """
    Manually trigger FL rounds
    
    Args:
        server_url: Server URL (e.g., http://localhost:5000)
        num_rounds: Number of rounds to run
        clients_per_round: Expected clients per round
    """
    print("="*70)
    print(f"Manual FL Round Trigger")
    print("="*70)
    print(f"Server: {server_url}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients per round: {clients_per_round}")
    print("="*70 + "\n")
    
    # Check server status
    print("🔍 Checking server status...")
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        status = response.json()
        print(f"✅ Server is running")
        print(f"   Current round: {status.get('current_round', 0)}")
        print(f"   Round in progress: {status.get('round_in_progress', False)}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Start rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"🚀 Starting Round {round_num}/{num_rounds}")
        print(f"{'='*70}")
        
        try:
            # Start round
            response = requests.post(f"{server_url}/start_round", timeout=5)
            result = response.json()
            
            if result.get('status') == 'started':
                print(f"✅ Round started")
                print(f"   Waiting for {clients_per_round} clients...")
                
                # Wait for completion
                while True:
                    time.sleep(2)
                    
                    status_response = requests.get(f"{server_url}/status", timeout=5)
                    status = status_response.json()
                    
                    if not status.get('round_in_progress', True):
                        print(f"\n✅ Round {round_num} completed!")
                        break
                    
                    submissions = status.get('submissions_received', 0)
                    print(f"   Progress: {submissions}/{clients_per_round}", end='\r')
                
                # Small delay
                if round_num < num_rounds:
                    print(f"⏳ Waiting 3s before next round...\n")
                    time.sleep(3)
            
            elif result.get('status') == 'completed':
                print("✅ All rounds already completed!")
                break
            
            elif result.get('status') == 'in_progress':
                print("⚠️  Round already in progress, waiting...")
                time.sleep(5)
            
            else:
                print(f"⚠️  Unexpected response: {result}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)
    
    # Show final metrics
    print("\n" + "="*70)
    print("📊 Final Metrics Summary")
    print("="*70)
    
    try:
        response = requests.get(f"{server_url}/metrics/summary", timeout=5)
        summary = response.json()
        
        print(f"\nRounds:")
        print(f"   Total: {summary['rounds']['total']}")
        print(f"   Avg Accuracy: {summary['rounds']['avg_accuracy']}%")
        print(f"   Max Accuracy: {summary['rounds']['max_accuracy']}%")
        
        print(f"\nClients:")
        print(f"   Total Submissions: {summary['clients']['total_submissions']}")
        print(f"   Avg Training Time: {summary['clients']['avg_training_time']}s")
    except:
        pass
    
    print("\n" + "="*70)
    print("✅ Done! Check dashboard at http://localhost:5001")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Manually trigger FL rounds')
    parser.add_argument('--server', type=str, default='http://localhost:5000', 
                       help='Server URL')
    parser.add_argument('--rounds', type=int, default=10, 
                       help='Number of rounds')
    parser.add_argument('--clients', type=int, default=5, 
                       help='Clients per round')
    
    args = parser.parse_args()
    
    trigger_rounds(args.server, args.rounds, args.clients)


if __name__ == "__main__":
    main()