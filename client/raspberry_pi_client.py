"""
Raspberry Pi Client Manager - SSH-based connection and remote execution
"""

import paramiko
import socket
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import threading
import subprocess


@dataclass
class RaspberryPiConfig:
    """Configuration for a Raspberry Pi client"""
    ip_address: str
    username: str
    password: str
    port: int = 22
    client_id: int = 0
    dataset_name: str = "MNIST"
    data_dir: str = "/home/pi/XFL-RPiLab/data"
    venv_path: str = "/home/pi/XFL-RPiLab/venv"
    project_path: str = "/home/pi/XFL-RPiLab"
    timeout: int = 30


class RaspberryPiConnection:
    """Manage SSH connection to a Raspberry Pi"""
    
    def __init__(self, config: RaspberryPiConfig):
        self.config = config
        self.client: Optional[paramiko.SSHClient] = None
        self.is_connected = False
        self._lock = threading.Lock()
        
    def connect(self, timeout: int = 30) -> bool:
        """Establish SSH connection to the Raspberry Pi"""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            print(f"🔌 Connecting to {self.config.username}@{self.config.ip_address}...")
            self.client.connect(
                hostname=self.config.ip_address,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout
            )
            self.is_connected = True
            print(f"✅ Connected to Raspberry Pi {self.config.ip_address}")
            return True
            
        except paramiko.AuthenticationException:
            print(f"❌ Authentication failed for {self.config.ip_address}")
            return False
        except socket.timeout:
            print(f"❌ Connection timeout for {self.config.ip_address}")
            return False
        except Exception as e:
            print(f"❌ Connection error for {self.config.ip_address}: {e}")
            return False
    
    def disconnect(self):
        """Close SSH connection"""
        with self._lock:
            if self.client:
                self.client.close()
                self.is_connected = False
                print(f"🔌 Disconnected from {self.config.ip_address}")
    
    def execute_command(self, command: str, timeout: int = 60) -> Tuple[str, str, int]:
        """
        Execute a command on the Raspberry Pi
        
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        if not self.is_connected or not self.client:
            return ("", "Not connected", 1)
        
        try:
            stdin, stdout, stderr = self.client.exec_command(
                command,
                timeout=timeout,
                get_pty=False
            )
            
            exit_code = stdout.channel.recv_exit_status()
            stdout_data = stdout.read().decode('utf-8', errors='ignore')
            stderr_data = stderr.read().decode('utf-8', errors='ignore')
            
            return stdout_data, stderr_data, exit_code
            
        except Exception as e:
            return ("", str(e), 1)
    
    def execute_command_streaming(self, command: str, timeout: int = 60):
        """
        Execute a command and yield output line by line
        """
        if not self.is_connected or not self.client:
            return
        
        try:
            channel = self.client.get_transport().open_session()
            channel.settimeout(timeout)
            channel.exec_command(command)
            
            while True:
                if channel.exit_status_ready():
                    break
                if channel.recv_ready():
                    yield channel.recv(1024).decode('utf-8', errors='ignore')
                time.sleep(0.1)
                    
        except Exception as e:
            yield f"Error: {e}"
    
    def check_system_info(self) -> Dict[str, Any]:
        """Get system information from the Raspberry Pi"""
        commands = {
            "hostname": "hostname",
            "uptime": "uptime -p",
            "cpu_info": "cat /proc/cpuinfo | grep 'model name' | head -1",
            "cpu_cores": "nproc",
            "memory_total": "free -m | awk '/Mem:/{print $2}'",
            "memory_available": "free -m | awk '/Mem:/{print $7}'",
            "disk_usage": "df -h / | awk 'NR==2{print $5}'",
            "temperature": "vcgencmd measure_temp 2>/dev/null || echo 'N/A'",
            "python_version": "python3 --version",
            "python_path": "which python3",
        }
        
        system_info = {}
        for key, cmd in commands.items():
            stdout, _, _ = self.execute_command(cmd, timeout=10)
            system_info[key] = stdout.strip()
        
        return system_info
    
    def check_project_exists(self) -> bool:
        """Check if the XFL-RPiLab project exists on the Raspberry Pi"""
        stdout, _, exit_code = self.execute_command(
            f"test -d {self.config.project_path} && echo 'exists' || echo 'missing'",
            timeout=10
        )
        return "exists" in stdout
    
    def setup_project(self) -> bool:
        """Setup the project on the Raspberry Pi (clone if needed)"""
        # Check if project exists
        if self.check_project_exists():
            print(f"📁 Project already exists on {self.config.ip_address}")
            return True
        
        # For now, assume project is already set up
        # In production, you would clone from a git repository
        print(f"⚠️ Project not found on {self.config.ip_address}")
        print(f"   Please manually setup the project at: {self.config.project_path}")
        return False
    
    def start_fl_client(
        self,
        server_url: str,
        local_epochs: int = 1,
        batch_size: int = 32,
        num_clients: int = 10
    ) -> bool:
        """Start the FL client on the Raspberry Pi"""
        
        activate_venv = f"source {self.config.venv_path}/bin/activate"
        
        command = f"""
            cd {self.config.project_path} && \
            {activate_venv} && \
            python -m client.run_client_standalone \
                --client-id {self.config.client_id} \
                --server-url {server_url} \
                --num-clients {self.config.num_clients} \
                --dataset {self.config.dataset_name} \
                --batch-size {batch_size} \
                --local-epochs {local_epochs} \
                --mode real-hardware
        """
        
        stdout, stderr, exit_code = self.execute_command(command, timeout=30)
        
        if exit_code == 0:
            print(f"✅ FL Client started on {self.config.ip_address}")
            return True
        else:
            print(f"❌ Failed to start FL Client: {stderr}")
            return False


class RaspberryPiManager:
    """Manage multiple Raspberry Pi clients"""
    
    def __init__(self, server_url: str = "http://192.168.100.68:5000"):
        self.connections: Dict[int, RaspberryPiConnection] = {}
        self.server_url = server_url
        
    def add_raspberry_pi(
        self,
        client_id: int,
        ip_address: str,
        username: str,
        password: str,
        **kwargs
    ) -> RaspberryPiConnection:
        """Add a Raspberry Pi to the manager"""
        config = RaspberryPiConfig(
            client_id=client_id,
            ip_address=ip_address,
            username=username,
            password=password,
            **kwargs
        )
        
        connection = RaspberryPiConnection(config)
        self.connections[client_id] = connection
        return connection
    
    def connect_all(self) -> Dict[int, bool]:
        """Connect to all Raspberry Pis"""
        results = {}
        for client_id, connection in self.connections.items():
            results[client_id] = connection.connect()
        return results
    
    def disconnect_all(self):
        """Disconnect from all Raspberry Pis"""
        for connection in self.connections.values():
            connection.disconnect()
    
    def get_system_info_all(self) -> Dict[int, Dict[str, Any]]:
        """Get system info from all connected Raspberry Pis"""
        results = {}
        for client_id, connection in self.connections.items():
            if connection.is_connected:
                results[client_id] = connection.check_system_info()
        return results
    
    def start_all_clients(
        self,
        local_epochs: int = 1,
        batch_size: int = 32,
        num_clients: int = 10
    ) -> Dict[int, bool]:
        """Start FL client on all Raspberry Pis"""
        results = {}
        for client_id, connection in self.connections.items():
            if connection.is_connected:
                results[client_id] = connection.start_fl_client(
                    server_url=self.server_url,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    num_clients=num_clients
                )
        return results


def create_raspberry_pi_client(
    ip_address: str = "192.168.100.41",
    username: str = "pi1",
    password: str = "1234",
    client_id: int = 0,
    server_url: str = "http://192.168.100.68:5000"
) -> RaspberryPiConnection:
    """Factory function to create a Raspberry Pi connection"""
    config = RaspberryPiConfig(
        client_id=client_id,
        ip_address=ip_address,
        username=username,
        password=password,
        dataset_name="MNIST",
        data_dir="/home/pi/XFL-RPiLab/data",
        venv_path="/home/pi/XFL-RPiLab/venv",
        project_path="/home/pi/XFL-RPiLab"
    )
    
    connection = RaspberryPiConnection(config)
    return connection


if __name__ == "__main__":
    # Test connection
    print("Testing Raspberry Pi connection...")
    rpi = create_raspberry_pi_client(
        ip_address="192.168.100.41",
        username="pi1",
        password="1234",
        client_id=0
    )
    
    if rpi.connect():
        print("\n📊 System Info:")
        info = rpi.check_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        rpi.disconnect()
    else:
        print("❌ Connection failed")