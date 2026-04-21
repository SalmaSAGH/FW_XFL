"""
Register Raspberry Pi as a physical client with the FL server
"""

import requests
import sys
import time

def register_physical_client(
    server_url: str = "http://192.168.100.68:5000",
    client_id: int = 0,
    ip_address: str = "192.168.100.41",
    hostname: str = "raspberry-pi-0",
    username: str = "pi1"
):
    """Register the Raspberry Pi as a physical client with the server"""
    
    print(f"Registering physical client {client_id} ({ip_address}) with server...")
    
    # Try the physical client registration endpoint
    try:
        response = requests.post(
            f"{server_url}/api/physical/register",
            json={
                "client_id": client_id,
                "ip_address": ip_address,
                "hostname": hostname,
                "username": username
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Registered successfully!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"⚠️ Registration endpoint not available (status {response.status_code})")
            print(f"   The server will need to be updated to support physical clients")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


def check_server_status(server_url: str = "http://192.168.100.68:5000"):
    """Check if server is running"""
    try:
        response = requests.get(f"{server_url}/api/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is running at {server_url}")
            return True
    except:
        pass
    print(f"❌ Server not reachable at {server_url}")
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register Raspberry Pi with FL server")
    parser.add_argument('--server-url', type=str, default='http://192.168.100.68:5000',
                        help='Server URL')
    parser.add_argument('--client-id', type=int, default=0,
                        help='Client ID for this Raspberry Pi')
    parser.add_argument('--ip', type=str, default='192.168.100.41',
                        help='Raspberry Pi IP address')
    parser.add_argument('--hostname', type=str, default='raspberry-pi-0',
                        help='Raspberry Pi hostname')
    parser.add_argument('--username', type=str, default='pi1',
                        help='Raspberry Pi username')
    
    args = parser.parse_args()
    
    # Check server status
    if not check_server_status(args.server_url):
        sys.exit(1)
    
    # Register client
    success = register_physical_client(
        server_url=args.server_url,
        client_id=args.client_id,
        ip_address=args.ip,
        hostname=args.hostname,
        username=args.username
    )
    
    sys.exit(0 if success else 1)