r"""
Quick deploy script - reads Raspberry Pi config from file and deploys auto-start
Usage: python scripts\quick_deploy.py --deploy
       python scripts\quick_deploy.py --check
       python scripts\quick_deploy.py --stop
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.auto_start_client import AutoStartSetup


def load_raspberry_pis(config_file: str):
    """Load Raspberry Pi configuration from file"""
    raspberry_pis = []
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return raspberry_pis
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    client_id = int(parts[0].strip())
                    ip = parts[1].strip()
                    username = parts[2].strip()
                    password = parts[3].strip()
                    raspberry_pis.append({
                        'client_id': client_id,
                        'ip_address': ip,
                        'username': username,
                        'password': password
                    })
                except ValueError:
                    print(f"⚠️ Invalid line: {line}")
    
    return raspberry_pis


def main():
    parser = argparse.ArgumentParser(description="Quick deploy auto-start to all Raspberry Pis")
    
    # Server configuration
    parser.add_argument('--server-ip', type=str, default='192.168.100.68',
                        help='Server IP address')
    parser.add_argument('--server-port', type=int, default=5000,
                        help='Server port')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset to use')
    parser.add_argument('--num-clients', type=int, default=40,
                        help='Total number of clients')
    parser.add_argument('--mode', type=str, default='real-hardware',
                        help='Metrics collection mode')
    parser.add_argument('--data-dir', type=str, default='/home/pi1/XFL-RPiLab/data',
                        help='Data directory on Raspberry Pi')
    # Get project root (parent of scripts directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(project_root, 'config', 'raspberry_pis.txt')
    
    parser.add_argument('--config', type=str, 
                        default=default_config,
                        help='Raspberry Pi config file')
    
    # Actions
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy auto-start service to all Raspberry Pis')
    parser.add_argument('--check', action='store_true',
                        help='Check service status on all Raspberry Pis')
    parser.add_argument('--stop', action='store_true',
                        help='Stop auto-start services on all Raspberry Pis')
    
    args = parser.parse_args()
    
    # Load Raspberry Pi configuration
    print(f"\n📂 Loading configuration from: {args.config}")
    raspberry_pis = load_raspberry_pis(args.config)
    
    if not raspberry_pis:
        print("❌ No Raspberry Pis configured")
        print(f"   Edit {args.config} to add your Raspberry Pis")
        return
    
    print(f"📱 Found {len(raspberry_pis)} Raspberry Pi(s):")
    for rpi in raspberry_pis:
        print(f"   Client {rpi['client_id']}: {rpi['ip_address']}")
    
    # Create setup
    setup = AutoStartSetup(
        server_ip=args.server_ip,
        server_port=args.server_port,
        dataset=args.dataset,
        num_clients=args.num_clients,
        mode=args.mode,
        data_dir=args.data_dir
    )
    
    # Add all Raspberry Pis
    for rpi in raspberry_pis:
        setup.add_raspberry_pi(
            client_id=rpi['client_id'],
            ip_address=rpi['ip_address'],
            username=rpi['username'],
            password=rpi['password']
        )
    
    # Connect to all Raspberry Pis
    print("\n🔌 Connecting to Raspberry Pis...")
    connection_results = setup.connect_all()
    
    for client_id, success in connection_results.items():
        status = "✅ Connected" if success else "❌ Failed"
        print(f"  Client {client_id}: {status}")
    
    # Perform action
    if args.deploy:
        results = setup.deploy_all()
        print("\n" + "="*60)
        print("📊 Deployment Results")
        print("="*60)
        success_count = sum(1 for s in results.values() if s)
        print(f"   Success: {success_count}/{len(results)}")
        for client_id, success in results.items():
            print(f"   Client {client_id}: {'✅' if success else '❌'}")
        
        print("\n" + "="*60)
        print("🚀 Next Steps")
        print("="*60)
        print("   1. Reboot each Raspberry Pi: sudo reboot")
        print("   2. The client will start automatically on boot")
        print("   3. Check status: python scripts\\quick_deploy.py --check")
        print("   4. View logs: sudo journalctl -u fl-client-ID -f")
    
    elif args.check:
        results = setup.check_services()
        print("\n" + "="*60)
        print("📊 Service Status")
        print("="*60)
        for client_id, info in results.items():
            status = info.get('status', 'unknown')
            print(f"   Client {client_id}: {status}")
    
    elif args.stop:
        results = setup.stop_services()
        print("\n" + "="*60)
        print("📊 Stop Results")
        print("="*60)
        for client_id, success in results.items():
            print(f"   Client {client_id}: {'✅ Stopped' if success else '❌ Failed'}")
    
    else:
        print("\n⚠️ No action specified.")
        print("   Use: --deploy, --check, or --stop")
        print("\nExamples:")
        print("   python scripts\\quick_deploy.py --deploy")
        print("   python scripts\\quick_deploy.py --check")
        print("   python scripts\\quick_deploy.py --stop")


if __name__ == "__main__":
    main()