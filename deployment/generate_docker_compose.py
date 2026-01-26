"""
Generate docker-compose.yml with N clients
Usage: python deployment/generate_docker_compose.py --num-clients 40
"""

import argparse
from pathlib import Path


def generate_docker_compose(num_clients: int, output_file: str = "docker-compose.yml"):
    """
    Generate docker-compose.yml with specified number of clients
    
    Args:
        num_clients: Number of client containers to create
        output_file: Output file path
    """
    
    compose_content = f"""version: '3.8'

services:
  # FL Server
  server:
    image: xfl-rpilab-client:latest
    container_name: xfl-server
    command: python -m server.run_server_standalone
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./server:/app/server
      - ./client:/app/client
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
      - NUM_CLIENTS={num_clients}
    networks:
      - fl-network

  # Dashboard
  dashboard:
    image: xfl-rpilab-client:latest
    container_name: xfl-dashboard
    command: python -m dashboard.run_dashboard_standalone
    ports:
      - "5001:5001"
    volumes:
      - ./logs:/app/logs
      - ./dashboard:/app/dashboard
    environment:
      - PYTHONUNBUFFERED=1
      - NUM_CLIENTS={num_clients}
    depends_on:
      - server
    networks:
      - fl-network

"""
    
    # Generate client services
    for i in range(num_clients):
        compose_content += f"""  # Client {i}
  client-{i}:
    image: xfl-rpilab-client:latest
    container_name: xfl-client-{i}
    command: python -m client.run_client_standalone --client-id {i} --server-host server --num-clients {num_clients} --batch-size 256
    volumes:
      - ./data:/app/data:ro
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - server
    networks:
      - fl-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1024M

"""
    
    # Add networks section
    compose_content += """networks:
  fl-network:
    driver: bridge
"""
    
    # Write to file
    output_path = Path(output_file)
    output_path.write_text(compose_content)
    
    print(f"✅ Generated {output_file} with {num_clients} clients")
    print(f"📁 File size: {output_path.stat().st_size} bytes")
    print(f"\n💡 To use this configuration:")
    print(f"   1. Build image: docker build -t xfl-rpilab-client .")
    print(f"   2. Start: docker-compose up")
    print(f"   3. Stop: docker-compose down")


def main():
    parser = argparse.ArgumentParser(description='Generate docker-compose.yml for XFL-RPiLab')
    parser.add_argument('--num-clients', type=int, default=5, help='Number of client containers')
    parser.add_argument('--output', type=str, default='docker-compose.yml', help='Output file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("XFL-RPiLab Docker Compose Generator")
    print("="*70)
    print(f"Generating configuration for {args.num_clients} clients...\n")
    
    generate_docker_compose(args.num_clients, args.output)


if __name__ == "__main__":
    main()