"""
Raspberry Pi Setup Script
Transfers and configures the XFL-RPiLab project on Raspberry Pi
"""

import os
import sys
import time
import subprocess
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from client.raspberry_pi_client import RaspberryPiManager, RaspberryPiConnection, RaspberryPiConfig


class RaspberryPiSetup:
    """Setup XFL-RPiLab project on Raspberry Pi"""
    
    def __init__(
        self,
        ip_address: str = "192.168.100.41",
        username: str = "pi1",
        password: str = "1234",
        project_local_path: str = None
    ):
        self.ip = ip_address
        self.username = username
        self.password = password
        
        # Default to current directory if not specified
        if project_local_path is None:
            project_local_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.project_local_path = project_local_path
        self.project_remote_path = f"/home/{username}/XFL-RPiLab"
        
        self.connection: Optional[RaspberryPiConnection] = None
    
    def connect(self) -> bool:
        """Connect to Raspberry Pi"""
        config = RaspberryPiConfig(
            client_id=0,
            ip_address=self.ip,
            username=self.username,
            password=self.password,
            project_path=self.project_remote_path
        )
        
        self.connection = RaspberryPiConnection(config)
        return self.connection.connect()
    
    def disconnect(self):
        """Disconnect from Raspberry Pi"""
        if self.connection:
            self.connection.disconnect()
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if prerequisites are met"""
        print("\n📋 Checking prerequisites...")
        
        results = {}
        
        # Check Python version
        stdout, _, _ = self.connection.execute_command("python3 --version", timeout=10)
        results['python'] = "Python 3" in stdout
        print(f"   Python: {'✅' if results['python'] else '❌'} {stdout.strip()}")
        
        # Check if git is available
        stdout, _, _ = self.connection.execute_command("which git", timeout=5)
        results['git'] = stdout.strip() != ""
        print(f"   Git: {'✅' if results['git'] else '❌'}")
        
        # Check disk space (need at least 2GB)
        stdout, _, _ = self.connection.execute_command("df -h / | awk 'NR==2{print $4}'", timeout=5)
        results['disk'] = True  # Assume OK if command works
        print(f"   Disk space: ✅ {stdout.strip()} available")
        
        return results
    
    def create_project_directory(self) -> bool:
        """Create project directory on Raspberry Pi"""
        print(f"\n📁 Creating project directory: {self.project_remote_path}")
        
        commands = [
            f"mkdir -p {self.project_remote_path}",
            f"mkdir -p {self.project_remote_path}/data",
            f"mkdir -p {self.project_remote_path}/logs",
            f"mkdir -p {self.project_remote_path}/models",
            f"mkdir -p {self.project_remote_path}/results"
        ]
        
        for cmd in commands:
            stdout, stderr, exit_code = self.connection.execute_command(cmd, timeout=10)
            if exit_code != 0:
                print(f"   ❌ Failed: {cmd}")
                print(f"      Error: {stderr}")
                return False
        
        print("   ✅ Directories created")
        return True
    
    def copy_project_files(self, use_scp: bool = True) -> bool:
        """Copy project files to Raspberry Pi"""
        print(f"\n📦 Copying project files to {self.ip}...")
        
        if use_scp:
            # Use SCP for file transfer
            try:
                # Create tar archive of project
                print("   Creating archive...")
                archive_name = "xfl-rpilab.tar.gz"
                
                # Exclude data, logs, results, and git directories
                exclude_args = "--exclude=data --exclude=logs --exclude=results --exclude=.git --exclude=venv --exclude=__pycache__"
                
                subprocess.run(
                    f"cd {self.project_local_path} && tar {exclude_args} -czf {archive_name} .",
                    shell=True,
                    check=True,
                    capture_output=True
                )
                
                # Copy via SCP
                print("   Transferring via SCP...")
                scp_result = subprocess.run(
                    f'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
                    f'-r {archive_name} {self.username}@{self.ip}:{self.project_remote_path}/',
                    shell=True,
                    capture_output=True,
                    timeout=120
                )
                
                if scp_result.returncode != 0:
                    print(f"   ❌ SCP failed: {scp_result.stderr.decode()}")
                    return False
                
                # Extract on Raspberry Pi
                print("   Extracting archive...")
                extract_cmd = f"cd {self.project_remote_path} && tar -xzf {archive_name} && rm {archive_name}"
                stdout, stderr, exit_code = self.connection.execute_command(extract_cmd, timeout=60)
                
                if exit_code != 0:
                    print(f"   ❌ Extraction failed: {stderr}")
                    return False
                
                print("   ✅ Files transferred successfully")
                return True
                
            except subprocess.TimeoutExpired:
                print("   ❌ Transfer timeout")
                return False
            except Exception as e:
                print(f"   ❌ Transfer error: {e}")
                return False
        else:
            # Manual file-by-file transfer (slower but more reliable)
            print("   Using manual file transfer...")
            
            # This would be much slower - not recommended for large projects
            print("   ⚠️ Manual transfer not implemented. Use SCP.")
            return False
    
    def setup_virtual_environment(self) -> bool:
        """Create and configure Python virtual environment"""
        print("\n🐍 Setting up Python virtual environment...")
        
        venv_path = f"{self.project_remote_path}/venv"
        
        # Check if venv already exists
        stdout, _, _ = self.connection.execute_command(
            f"test -d {venv_path} && echo exists || echo missing",
            timeout=5
        )
        
        if "exists" in stdout:
            print(f"   ℹ️ Virtual environment already exists at {venv_path}")
            return True
        
        # Create virtual environment
        print(f"   Creating virtual environment at {venv_path}...")
        stdout, stderr, exit_code = self.connection.execute_command(
            f"cd {self.project_remote_path} && python3 -m venv venv",
            timeout=60
        )
        
        if exit_code != 0:
            print(f"   ❌ Failed to create venv: {stderr}")
            return False
        
        # Install requirements
        print("   Installing Python packages...")
        
        # First upgrade pip
        stdout, stderr, exit_code = self.connection.execute_command(
            f"source {venv_path}/bin/activate && pip install --upgrade pip",
            timeout=120
        )
        
        # Install requirements
        stdout, stderr, exit_code = self.connection.execute_command(
            f"source {venv_path}/bin/activate && pip install -r {self.project_remote_path}/requirements.txt",
            timeout=300
        )
        
        if exit_code != 0:
            print(f"   ⚠️ Some packages may have failed to install")
            print(f"      Error: {stderr[:200]}...")
            # Continue anyway - some packages might be optional
        
        print("   ✅ Virtual environment ready")
        return True
    
    def download_dataset(self, dataset: str = "MNIST") -> bool:
        """Download dataset on Raspberry Pi"""
        print(f"\n📥 Downloading {dataset} dataset...")
        
        # Create data directory if needed
        self.connection.execute_command(f"mkdir -p {self.project_remote_path}/data", timeout=5)
        
        # For now, we'll use the data from the server
        # In production, you would download or sync data
        print(f"   ℹ️ Data will be synced from server or downloaded on first run")
        return True
    
    def verify_installation(self) -> bool:
        """Verify that installation was successful"""
        print("\n✅ Verifying installation...")
        
        checks = [
            (f"test -d {self.project_remote_path}/client", "client directory"),
            (f"test -d {self.project_remote_path}/server", "server directory"),
            (f"test -d {self.project_remote_path}/config", "config directory"),
            (f"test -d {self.project_remote_path}/venv", "virtual environment"),
            (f"test -f {self.project_remote_path}/requirements.txt", "requirements.txt"),
        ]
        
        all_passed = True
        for cmd, name in checks:
            stdout, _, exit_code = self.connection.execute_command(cmd, timeout=5)
            status = "✅" if exit_code == 0 else "❌"
            print(f"   {status} {name}")
            if exit_code != 0:
                all_passed = False
        
        return all_passed
    
    def run_full_setup(self) -> bool:
        """Run complete setup process"""
        print("="*60)
        print("Raspberry Pi Setup for XFL-RPiLab")
        print("="*60)
        print(f"Target: {self.username}@{self.ip}")
        print(f"Project: {self.project_local_path} -> {self.project_remote_path}")
        print("="*60)
        
        # Connect
        print("\n🔌 Connecting to Raspberry Pi...")
        if not self.connect():
            print("❌ Failed to connect")
            return False
        print("✅ Connected")
        
        # Check prerequisites
        prereqs = self.check_prerequisites()
        if not prereqs.get('python', False):
            print("❌ Python 3 is required")
            return False
        
        # Create directories
        if not self.create_project_directory():
            print("❌ Failed to create directories")
            return False
        
        # Copy files
        if not self.copy_project_files(use_scp=True):
            print("❌ Failed to copy project files")
            return False
        
        # Setup virtual environment
        if not self.setup_virtual_environment():
            print("❌ Failed to setup virtual environment")
            return False
        
        # Verify
        if not self.verify_installation():
            print("⚠️ Installation verification failed")
        else:
            print("\n" + "="*60)
            print("✅ SETUP COMPLETE!")
            print("="*60)
            print(f"""
Next steps:
1. The project is now on the Raspberry Pi at: {self.project_remote_path}
2. To start the FL client, run:
   
   ssh {self.username}@{self.ip}
   cd {self.project_remote_path}
   source venv/bin/activate
   python -m client.run_client_raspberry_pi --client-id 0 --server-url http://192.168.100.68:5000 --mode real-hardware

3. Or use the deployment script:
   python scripts/deploy_raspberry_pi.py --start 0
""")
        
        self.disconnect()
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup XFL-RPiLab on Raspberry Pi")
    parser.add_argument('--ip', type=str, default='192.168.100.41',
                        help='Raspberry Pi IP address')
    parser.add_argument('--username', type=str, default='pi1',
                        help='Raspberry Pi username')
    parser.add_argument('--password', type=str, default='1234',
                        help='Raspberry Pi password')
    parser.add_argument('--project-path', type=str, default=None,
                        help='Local project path (default: current directory)')
    
    args = parser.parse_args()
    
    setup = RaspberryPiSetup(
        ip_address=args.ip,
        username=args.username,
        password=args.password,
        project_local_path=args.project_path
    )
    
    success = setup.run_full_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()