"""
Sync code to Raspberry Pis
Updates the Python source code on all configured Raspberry Pis
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.auto_start_client import AutoStartSetup


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent.absolute()


def sync_code_to_raspberry_pi(connection, rpi_config: dict) -> bool:
    """Sync the project code to a Raspberry Pi"""
    client_id = rpi_config['client_id']
    ip = rpi_config['ip_address']
    
    print(f"\n📦 Syncing code to Client {client_id} ({ip})...")
    
    # Key directories/files to sync (excluding data, logs, venv, etc.)
    sync_items = [
        'client/',
        'server/',
        'config/',
        'scripts/',
        'requirements.txt',
        'requirements_server.txt',
    ]
    
    project_root = get_project_root()
    remote_path = rpi_config['project_path']
    
    for item in sync_items:
        local_path = project_root / item
        if local_path.exists():
            print(f"   📁 Syncing {item}...")
            # Use rsync if available, otherwise use scp
            cmd = f"rsync -avz --progress -e ssh {local_path}/ pi1@{ip}:{remote_path}/{item}/"
            stdout, stderr, exit_code = connection.execute_command(
                f"rsync -avz -e ssh {local_path}/ pi1@{ip}:{remote_path}/{item}/",
                timeout=120
            )
            if exit_code != 0:
                # Fallback to scp
                print(f"   ⚠️ rsync failed, trying scp...")
                scp_cmd = f"scp -r {local_path}/* pi1@{ip}:{remote_path}/{item}/"
                stdout, stderr, exit_code = connection.execute_command(scp_cmd, timeout=120)
                if exit_code != 0:
                    print(f"   ❌ Failed to sync {item}")
                    return False
    
    print(f"   ✅ Code synced successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync code to Raspberry Pis")
    parser.add_argument('--config', type=str, 
                        default='config/raspberry_pis.txt',
                        help='Raspberry Pi config file')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = get_project_root()
    config_file = project_root / args.config
    
    # Load Raspberry Pi configuration
    print(f"\n📂 Loading configuration from: {config_file}")
    
    # Simple config loader
    raspberry_pis = []
    if config_file.exists():
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    raspberry_pis.append({
                        'client_id': int(parts[0].strip()),
                        'ip_address': parts[1].strip(),
                        'username': parts[2].strip(),
                        'password': parts[3].strip(),
                        'project_path': f"/home/{parts[2].strip()}/XFL-RPiLab"
                    })
    
    if not raspberry_pis:
        print("❌ No Raspberry Pis configured")
        return
    
    print(f"📱 Found {len(raspberry_pis)} Raspberry Pi(s)")
    
    # Create setup
    setup = AutoStartSetup(
        server_ip='192.168.100.68',
        server_port=5000
    )
    
    for rpi in raspberry_pis:
        setup.add_raspberry_pi(
            client_id=rpi['client_id'],
            ip_address=rpi['ip_address'],
            username=rpi['username'],
            password=rpi['password']
        )
    
    # Connect
    print("\n🔌 Connecting to Raspberry Pis...")
    connection_results = setup.connect_all()
    
    # Sync code to each Raspberry Pi
    for rpi_config in setup.raspberry_pis:
        client_id = rpi_config['client_id']
        connection = setup.rpi_manager.connections.get(client_id)
        
        if connection and connection.is_connected:
            sync_code_to_raspberry_pi(connection, rpi_config)
        else:
            print(f"\n❌ Client {client_id}: Not connected, skipping")
    
    # Disconnect
    setup.rpi_manager.disconnect_all()
    
    print("\n✅ Code sync complete!")
    print("\n📌 Next: Restart the services to use new code:")
    print("   ssh pi1@192.168.100.41 'sudo systemctl restart fl-client-0'")
    print("   ssh pi1@192.168.100.40 'sudo systemctl restart fl-client-1'")


if __name__ == "__main__":
    main()