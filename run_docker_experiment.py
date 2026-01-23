"""
Run FL experiment with Docker clients
Automates the entire process: build, deploy, monitor
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(command, shell=True):
    """Execute a shell command"""
    print(f"\n🔧 Executing: {command}")
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         XFL-RPiLab Docker Experiment Launcher                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
    except:
        print("❌ Error: Docker or docker-compose not found!")
        print("   Please install Docker Desktop for Windows")
        sys.exit(1)
    
    print("✅ Docker is installed\n")
    
    # Ask for number of clients
    try:
        num_clients = int(input("How many clients do you want to simulate? (5-40): "))
        if num_clients < 1 or num_clients > 40:
            print("⚠️  Using default: 5 clients")
            num_clients = 5
    except:
        print("⚠️  Invalid input, using default: 5 clients")
        num_clients = 5
    
    print(f"\n📊 Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Server: localhost:5000")
    print(f"   Dashboard: localhost:5001")
    
    # Ask if user wants to clean old data
    try:
        clean = input("\n🧹 Clean old experiment data? (y/n): ").lower()
        if clean == 'y':
            db_file = Path("logs/server_metrics.db")
            if db_file.exists():
                db_file.unlink()
                print("✅ Old database deleted")
            else:
                print("ℹ️  No old database found")
    except:
        pass
    
    # Generate docker-compose.yml
    print("\n" + "="*70)
    print("Step 1: Generating docker-compose.yml")
    print("="*70)
    
    run_command(f"python deployment/generate_docker_compose.py --num-clients {num_clients}")
    
    # Build Docker image
    print("\n" + "="*70)
    print("Step 2: Building Docker image (this may take a few minutes...)")
    print("="*70)
    
    run_command("docker build -t xfl-rpilab-client .")
    
    print("\n✅ Docker image built successfully!")
    
    # Start containers
    print("\n" + "="*70)
    print("Step 3: Starting containers")
    print("="*70)
    
    print("\n💡 Containers are starting...")
    print("   - Server will be available at: http://localhost:5000")
    print("   - Dashboard will be available at: http://localhost:5001")
    print(f"   - {num_clients} clients will connect automatically")
    
    run_command("docker-compose up -d")
    
    print("\n⏳ Waiting for containers to initialize (10 seconds)...")
    time.sleep(10)
    
    # Show status
    print("\n" + "="*70)
    print("Container Status")
    print("="*70)
    run_command("docker-compose ps")
    
    print("\n" + "="*70)
    print("✅ Experiment Started Successfully!")
    print("="*70)
    
    print(f"\n📊 What's happening now:")
    print(f"   • Server is running and waiting for rounds to start")
    print(f"   • {num_clients} clients are connected and waiting")
    print(f"   • Dashboard is live at http://localhost:5001")
    
    print(f"\n🔧 To start FL rounds:")
    print(f"   You need to trigger rounds via the server API")
    print(f"   The clients will automatically participate")
    
    print(f"\n📝 Useful commands:")
    print(f"   • View logs: docker-compose logs -f [service-name]")
    print(f"   • Stop: docker-compose down")
    print(f"   • Restart: docker-compose restart")
    
    print(f"\n💡 Open http://localhost:5001 in your browser to see the dashboard!")
    
    # Ask if user wants to see logs
    try:
        follow_logs = input("\n📋 Do you want to follow the logs? (y/n): ").lower()
        if follow_logs == 'y':
            print("\n🔄 Following logs (Press Ctrl+C to stop)...")
            run_command("docker-compose logs -f")
    except KeyboardInterrupt:
        print("\n\n✅ Stopped following logs")
    
    print("\n" + "="*70)
    print("To stop the experiment: docker-compose down")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()