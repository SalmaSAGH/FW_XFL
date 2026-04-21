"""
Auto-Start Client Setup for Raspberry Pi
Creates and deploys a systemd service that automatically starts the FL client on boot
"""

import os
import sys
import argparse
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.raspberry_pi_client import RaspberryPiManager, RaspberryPiConnection, RaspberryPiConfig


# Systemd service template
SYSTEMD_SERVICE_TEMPLATE = """[Unit]
Description=XFL-RPiLab FL Client for {client_id}
After=network.target

[Service]
Type=simple
User={username}
WorkingDirectory={project_path}
Environment="PATH={venv_path}/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH={project_path}"
ExecStart={venv_path}/bin/python {project_path}/client/run_client_raspberry_pi.py \\
    --client-id {client_id} \\
    --server-host {server_ip} \\
    --server-port {server_port} \\
    --dataset {dataset} \\
    --num-clients {num_clients} \\
    --mode {mode} \\
    --data-dir {data_dir}
Restart=always
RestartSec=10

# Logging
StandardOutput=append:{log_path}/client_{client_id}.out.log
StandardError=append:{log_path}/client_{client_id}.err.log

[Install]
WantedBy=multi-user.target
"""

# systemd service for auto-registration only (lighter weight)
SYSTEMD_SERVICE_LIGHT_TEMPLATE = """[Unit]
Description=XFL-RPiLab FL Client for {client_id}
After=network.target

[Service]
Type=simple
User={username}
WorkingDirectory={project_path}
Environment="PATH={venv_path}/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart={venv_path}/bin/python {project_path}/client/run_client_raspberry_pi.py \\
    --client-id {client_id} \\
    --server-host {server_ip} \\
    --server-port {server_port} \\
    --dataset {dataset} \\
    --num-clients {num_clients} \\
    --mode {mode} \\
    --data-dir {data_dir}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""


class AutoStartSetup:
    """Setup auto-start systemd service on Raspberry Pi"""
    
    def __init__(
        self,
        server_ip: str = "192.168.100.68",
        server_port: int = 5000,
        dataset: str = "MNIST",
        num_clients: int = 40,
        mode: str = "real-hardware",
        data_dir: str = "/home/pi1/XFL-RPiLab/data"
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_url = f"http://{server_ip}:{server_port}"
        self.dataset = dataset
        self.num_clients = num_clients
        self.mode = mode
        self.data_dir = data_dir
        
        self.rpi_manager = RaspberryPiManager(server_url=self.server_url)
        self.raspberry_pis: List[Dict] = []
    
    def add_raspberry_pi(
        self,
        client_id: int,
        ip_address: str,
        username: str = "pi1",
        password: str = "1234",
        project_path: str = None,
        venv_path: str = None
    ):
        """Add a Raspberry Pi to configure"""
        if project_path is None:
            project_path = f"/home/{username}/XFL-RPiLab"
        if venv_path is None:
            venv_path = f"/home/{username}/XFL-RPiLab/venv"
            
        self.raspberry_pis.append({
            'client_id': client_id,
            'ip_address': ip_address,
            'username': username,
            'password': password,
            'project_path': project_path,
            'venv_path': venv_path
        })
        
        # Add to manager
        self.rpi_manager.add_raspberry_pi(
            client_id=client_id,
            ip_address=ip_address,
            username=username,
            password=password,
            project_path=project_path,
            venv_path=venv_path
        )
    
    def connect_all(self) -> Dict[int, bool]:
        """Connect to all Raspberry Pis"""
        print("\n" + "="*60)
        print("Connecting to Raspberry Pis...")
        print("="*60)
        return self.rpi_manager.connect_all()
    
    def create_service_file(self, rpi_config: Dict) -> str:
        """Create systemd service file content"""
        return SYSTEMD_SERVICE_TEMPLATE.format(
            client_id=rpi_config['client_id'],
            username=rpi_config['username'],
            project_path=rpi_config['project_path'],
            venv_path=rpi_config['venv_path'],
            server_ip=self.server_ip,
            server_port=self.server_port,
            dataset=self.dataset,
            num_clients=self.num_clients,
            mode=self.mode,
            data_dir=self.data_dir,
            log_path=f"{rpi_config['project_path']}/logs"
        )
    
    def deploy_to_raspberry_pi(self, rpi_config: Dict) -> bool:
        """Deploy and enable auto-start service on a Raspberry Pi"""
        client_id = rpi_config['client_id']
        connection = self.rpi_manager.connections.get(client_id)
        
        if not connection or not connection.is_connected:
            print(f"  ❌ Client {client_id}: Not connected")
            return False
        
        print(f"\n  📦 Deploying to Client {client_id}...")
        
        # Step 1: Create service file
        service_name = f"fl-client-{client_id}.service"
        service_content = self.create_service_file(rpi_config)
        service_remote_path = f"/tmp/{service_name}"
        
        # Write service file locally first
        local_service_path = f"/tmp/{service_name}"
        with open(local_service_path, 'w') as f:
            f.write(service_content)
        
        # Upload service file using SFTP
        print(f"    📤 Uploading service file...")
        try:
            sftp = connection.client.open_sftp()
            sftp.put(local_service_path, service_remote_path)
            sftp.close()
            upload_success = True
        except Exception as e:
            print(f"    ❌ Failed to upload service file: {e}")
            return False
        
        # Step 2: Move to systemd directory (requires sudo with -S for password)
        print(f"    🔧 Installing systemd service...")
        password = rpi_config.get('password', '')
        
        # Use echo to pipe password to sudo -S
        sudo_commands = [
            f"echo '{password}' | sudo -S mv {service_remote_path} /etc/systemd/system/",
            f"echo '{password}' | sudo -S chmod 644 /etc/systemd/system/{service_name}",
            f"echo '{password}' | sudo -S systemctl daemon-reload",
            f"echo '{password}' | sudo -S systemctl enable {service_name}",
            f"echo '{password}' | sudo -S systemctl start {service_name}"
        ]
        
        for cmd in sudo_commands:
            stdout, stderr, exit_code = connection.execute_command(cmd, timeout=30)
            if exit_code != 0:
                print(f"    ❌ Command failed: {cmd}")
                print(f"       Error: {stderr}")
                return False
        
        # Step 3: Verify service is running
        print(f"    ✅ Verifying service status...")
        stdout, stderr, exit_code = connection.execute_command(
            f"echo '{password}' | sudo -S systemctl status {service_name}", timeout=10
        )
        
        if exit_code == 0:
            print(f"    ✅ Service is running!")
            return True
        else:
            print(f"    ⚠️ Service may not be running yet (this is normal on first start)")
            print(f"       Check logs with: sudo journalctl -u {service_name} -f")
            return True
    
    def deploy_all(self) -> Dict[int, bool]:
        """Deploy auto-start to all configured Raspberry Pis"""
        print("\n" + "="*60)
        print("Deploying Auto-Start Services")
        print("="*60)
        
        results = {}
        for rpi_config in self.raspberry_pis:
            client_id = rpi_config['client_id']
            ip = rpi_config['ip_address']
            
            print(f"\n🔄 Processing Client {client_id} ({ip})...")
            success = self.deploy_to_raspberry_pi(rpi_config)
            results[client_id] = success
        
        return results
    
    def check_services(self) -> Dict[int, Dict]:
        """Check status of auto-start services"""
        print("\n" + "="*60)
        print("Checking Service Status")
        print("="*60)
        
        results = {}
        for rpi_config in self.raspberry_pis:
            client_id = rpi_config['client_id']
            connection = self.rpi_manager.connections.get(client_id)
            
            if not connection or not connection.is_connected:
                results[client_id] = {'status': 'offline'}
                continue
            
            service_name = f"fl-client-{client_id}.service"
            stdout, stderr, exit_code = connection.execute_command(
                f"sudo systemctl is-active {service_name}", timeout=10
            )
            
            status = "active" if "active" in stdout else "inactive"
            results[client_id] = {'status': status, 'output': stdout.strip()}
            
            print(f"  Client {client_id}: {status}")
        
        return results
    
    def stop_services(self) -> Dict[int, bool]:
        """Stop auto-start services"""
        print("\n" + "="*60)
        print("Stopping Auto-Start Services")
        print("="*60)
        
        results = {}
        for rpi_config in self.raspberry_pis:
            client_id = rpi_config['client_id']
            connection = self.rpi_manager.connections.get(client_id)
            
            if not connection or not connection.is_connected:
                results[client_id] = False
                continue
            
            service_name = f"fl-client-{client_id}.service"
            stdout, stderr, exit_code = connection.execute_command(
                f"sudo systemctl stop {service_name}", timeout=10
            )
            
            results[client_id] = (exit_code == 0)
            print(f"  Client {client_id}: {'✅ Stopped' if results[client_id] else '❌ Failed'}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Setup auto-start FL client on Raspberry Pi")
    
    # Server configuration
    parser.add_argument('--server-ip', type=str, default='192.168.100.68',
                        help='Server IP address')
    parser.add_argument('--server-port', type=int, default=5000,
                        help='Server port')
    
    # Client configuration
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset to use')
    parser.add_argument('--num-clients', type=int, default=40,
                        help='Total number of clients')
    parser.add_argument('--mode', type=str, default='real-hardware',
                        choices=['simulated', 'real-hardware'],
                        help='Metrics collection mode')
    parser.add_argument('--data-dir', type=str, default='/home/pi1/XFL-RPiLab/data',
                        help='Data directory on Raspberry Pi')
    
    # Raspberry Pi configuration (can be specified multiple times)
    parser.add_argument('--add-rpi', action='append', nargs=4, metavar=('ID', 'IP', 'USER', 'PASS'),
                        help='Add Raspberry Pi: --add-rpi CLIENT_ID IP USERNAME PASSWORD')
    
    # Actions
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy auto-start service to all Raspberry Pis')
    parser.add_argument('--check', action='store_true',
                        help='Check service status on all Raspberry Pis')
    parser.add_argument('--stop', action='store_true',
                        help='Stop auto-start services on all Raspberry Pis')
    
    args = parser.parse_args()
    
    # Create setup
    setup = AutoStartSetup(
        server_ip=args.server_ip,
        server_port=args.server_port,
        dataset=args.dataset,
        num_clients=args.num_clients,
        mode=args.mode,
        data_dir=args.data_dir
    )
    
    # Add Raspberry Pis from command line
    if args.add_rpi:
        for rpi_info in args.add_rpi:
            client_id = int(rpi_info[0])
            ip = rpi_info[1]
            username = rpi_info[2]
            password = rpi_info[3]
            setup.add_raspberry_pi(client_id, ip, username, password)
            print(f"Added: Client {client_id} at {ip}")
    
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
        print("Deployment Results")
        print("="*60)
        for client_id, success in results.items():
            print(f"  Client {client_id}: {'✅ Success' if success else '❌ Failed'}")
    
    elif args.check:
        results = setup.check_services()
    
    elif args.stop:
        results = setup.stop_services()
    
    else:
        print("\n⚠️ No action specified. Use --deploy, --check, or --stop")
        print("\nExample usage:")
        print("  python scripts/auto_start_client.py \\")
        print("    --server-ip 192.168.100.68 \\")
        print("    --add-rpi 0 192.168.100.41 p1 1234 \\")
        print("    --add-rpi 1 192.168.100.40 p1 1234 \\")
        print("    --deploy")


if __name__ == "__main__":
    main()