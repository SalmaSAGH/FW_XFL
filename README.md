# XFL-RPiLab: Federated Learning with Physical Raspberry Pi Clients

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)

A sophisticated federated learning framework designed for real-world deployment using physical Raspberry Pi devices as edge clients. This system enables distributed machine learning across resource-constrained devices while maintaining privacy and providing comprehensive monitoring capabilities.

## 🌟 Key Features

### 🔬 Federated Learning Core
- **Multi-Dataset Support**: MNIST, CIFAR-10, CIFAR-100, FashionMNIST, EMNIST
- **Flexible Model Architectures**: SimpleCNN, DepthwiseCNN, CIFAR100CNN, MobileNetV2, ResNet8, ShuffleNetV2
- **XFL Strategies**: Exchangeable Federated Learning with configurable layer sharing
- **Real-time Aggregation**: FedAvg and XFL aggregation strategies
- **Privacy-Preserving**: Model weights never leave client devices

### 🖥️ Physical Client Management
- **Raspberry Pi Integration**: Native support for physical RPi devices
- **Heartbeat Monitoring**: Real-time client health and connectivity tracking
- **Automatic Client Discovery**: Dynamic registration and management
- **Network Quality Assessment**: Latency, packet loss, and jitter monitoring
- **Resource Constraints**: CPU, RAM, and power consumption tracking

### 📊 Monitoring & Analytics
- **Real-time Dashboard**: Web-based monitoring interface
- **Comprehensive Metrics**: Training loss, accuracy, latency, energy consumption
- **Historical Analysis**: Round-by-round performance tracking
- **Client Performance**: Individual device metrics and diagnostics
- **Database Integration**: PostgreSQL for persistent metric storage

### 🐳 Containerized Deployment
- **Docker Compose**: Simplified multi-service deployment
- **Physical vs Virtual**: Separate configurations for real hardware and simulation
- **Scalable Architecture**: Easy addition of new clients and services
- **Development Environment**: Isolated development with hot-reload

### 🔧 Advanced Features
- **Configurable Training**: Dynamic hyperparameters (learning rate, epochs, batch size)
- **Data Distribution**: IID and non-IID data partitioning strategies
- **Model Compression**: Quantization and sparsification support
- **Fault Tolerance**: Automatic retry mechanisms and timeout handling
- **RESTful API**: Comprehensive API for external integrations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │     Server      │    │   Dashboard     │
│   (React/Vite)  │◄──►│   (Flask)       │◄──►│   (Flask)       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RPi Client 1  │    │   RPi Client 2  │    │   RPi Client N  │
│   (Python)      │    │   (Python)      │    │   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

- **Server**: Central FL orchestration, client management, aggregation
- **Clients**: Edge devices performing local training
- **Dashboard**: Monitoring and visualization interface
- **Frontend**: User interface for configuration and control
- **Database**: Metrics and experiment data storage

## 📋 Prerequisites

### Hardware Requirements
- **Server**: Modern CPU with 4+ cores, 8GB+ RAM, 50GB storage
- **Clients**: Raspberry Pi 4B+ (4GB RAM recommended), WiFi connectivity
- **Network**: Stable WiFi network for client-server communication

### Software Requirements
- **Docker & Docker Compose**: Latest versions
- **Python**: 3.8 or higher
- **Git**: For cloning the repository
- **Web Browser**: Modern browser for dashboard access

### Raspberry Pi Setup
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.8+ with pip
- **Dependencies**: As specified in `requirements.txt`
- **Network**: Static IP or DHCP reservation recommended

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/XFL-RPiLab.git
cd XFL-RPiLab
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Docker Deployment
```bash
# For physical Raspberry Pi clients
docker-compose -f docker-compose-physical.yml up --build -d

# For virtual client simulation
docker-compose -f docker-compose.yml up --build -d
```

### 4. Raspberry Pi Client Setup
```bash
# On each Raspberry Pi
git clone https://github.com/SalmaSAGH/FW_XFL.git
cd XFL-RPiLab

# Install client dependencies
pip install -r requirements.txt

# Configure client
nano config/raspberry_pis.txt  # Add Pi IP addresses
python scripts/setup_raspberry_pi.py
```

## ⚙️ Configuration

### Server Configuration
Edit `config/config.yaml`:
```yaml
dataset:
  name: "CIFAR10"
  data_distribution: "iid"

model:
  name: "DepthwiseCNN"
  num_classes: 10

training:
  num_rounds: 100
  clients_per_round: 5
  local_epochs: 1
  batch_size: 32
  learning_rate: 0.01

network:
  simulate_constraints: false
  latency_ms: 50
  bandwidth_mbps: 10
```

### Client Configuration
Modify `client/run_client_raspberry_pi.py`:
```python
# Client ID and server URL
CLIENT_ID = 1
SERVER_URL = "http://192.168.1.100:5000"
```

### Physical Client Registration
Update `config/raspberry_pis.txt`:
```
192.168.1.101
192.168.1.102
192.168.1.103
192.168.1.104
192.168.1.105
```

## 🎯 Usage

### Starting a Federated Learning Experiment

1. **Launch Services**:
```bash
docker-compose -f docker-compose-physical.yml up -d
```

2. **Access Dashboard**:
   - Open http://localhost:5001 in your browser
   - Monitor real-time training progress

3. **Trigger FL Round**:
```bash
python deployment/trigger_fl_rounds.py
```

4. **Monitor Clients**:
   - Dashboard shows client status and metrics
   - Logs available via `docker logs xfl-server`

### Manual Client Operation

```bash
# On Raspberry Pi
python -m client.run_client_raspberry_pi --client-id 1
```

### API Usage

```python
import requests

# Get server status
response = requests.get("http://localhost:5000/api/status")
status = response.json()

# Start new round
requests.post("http://localhost:5000/api/start_round")

# Get global model
model_data = requests.get("http://localhost:5000/api/get_global_model").json()
```

## 📡 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get server status and round information |
| `/api/start_round` | POST | Initiate new FL round |
| `/api/get_global_model` | GET | Download current global model |
| `/api/submit_update` | POST | Submit client model update |
| `/api/physical/heartbeat` | POST | Client heartbeat registration |

### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics/summary` | GET | Get aggregated metrics |
| `/api/physical/clients` | GET | List registered physical clients |
| `/api/physical/config` | GET | Get client configuration |

### XFL Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/xfl/set_strategy` | POST | Configure XFL strategy |
| `/api/xfl/get_strategy` | GET | Get current XFL configuration |

## 🔧 Development

### Project Structure
```
XFL-RPiLab/
├── client/                 # Client-side code
│   ├── client.py          # Main FL client logic
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # Model architectures
│   └── run_client_raspberry_pi.py
├── server/                 # Server-side code
│   ├── server.py          # Main FL server
│   ├── physical_client_manager.py
│   └── run_server_standalone.py
├── dashboard/             # Monitoring dashboard
│   ├── dashboard.py
│   └── run_dashboard_standalone.py
├── frontend/              # React web interface
│   ├── src/
│   └── package.json
├── config/                # Configuration files
├── deployment/            # Deployment scripts
├── scripts/               # Utility scripts
└── docker-compose*.yml    # Docker configurations
```

### Adding New Models

1. Define model in `client/model.py`:
```python
class NewModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Model definition

    def forward(self, x):
        # Forward pass
```

2. Register in `MODEL_CONFIG`:
```python
MODEL_CONFIG = {
    'NewModel': (num_classes, in_channels, input_size),
}
```

### Custom Datasets

1. Add dataset class in `client/dataset.py`
2. Register in `DATASET_CONFIG`
3. Update transforms if needed

### Testing

```bash
# Run unit tests
python -m pytest

# Test specific components
python client/test_model.py
python experiments/experiment.py
```

## 📈 Performance Optimization

### For Raspberry Pi Clients
- Use `DepthwiseCNN` for CIFAR datasets (efficient on ARM)
- Reduce `local_epochs` to 1 for faster rounds
- Enable quantization for reduced bandwidth
- Monitor CPU/RAM usage via dashboard

### Network Optimization
- Stable WiFi connection essential
- Reduce model size with XFL strategies
- Implement retry mechanisms for unreliable networks
- Use compression for large model updates

### Server Optimization
- PostgreSQL for metric storage
- Asynchronous client handling
- Configurable timeouts and batch sizes
- Resource limits in Docker Compose

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on PyTorch and Flask frameworks
- Inspired by federated learning research
- Raspberry Pi community for hardware support
- Open-source contributors

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the [Wiki](wiki) for detailed guides
- Review the [FAQ](docs/FAQ.md) for common questions

---

**Happy Federated Learning! 🚀**
```
RPi: sudo apt autoremove && sudo apt clean
sudo raspi-config → Advanced → Expand Filesystem → Reboot
df -h  # Check / free space
```

**Ports (PC):**
- 🖥️ Dashboard: http://localhost:5001
- 🔌 API/Server: http://localhost:5000 (RPi connects to 192.168.100.68:5000)
- 🌐 Frontend: http://localhost:3000

### 2️⃣ Scale to Multiple RPis
```
# Update config.yaml: num_clients=N, clients_per_round=M
# On each RPi: bash rpi_client_run.sh --client-id X (unique 0,1,2...)
```

### 3️⃣ Original Docker Simulation (10 containers)
```
python deployment/generate_docker_compose.py --num-clients 10
docker compose up  # Simulated clients (legacy)
```

### RPi Metrics (New! 🔥)
- 📊 CPU/RAM/psutil (cross-platform)
- 🌡️ **RPi Temp/Volts/Throttle** (vcgencmd)
- ⚡ Energy estimation (realistic RPi model)
- 🌐 Real WiFi network (latency/jitter/throughput)

**Example Dashboard Metrics:**
```
Client 0 (RPi 192.168.100.41):
├── CPU: 45% | RAM: 128MB | Temp: 52.3°C
├── Training Acc: 92.1% | Time: 12.4s
├── Network: 1.2MB ↑ | Latency: 28ms | WiFi Throughput: 15Mbps
└── Energy: 0.024Wh | Power: 2.8W
```

## 📋 Full File Structure (Post-Migration)
```
XFL-RPiLab/
├── docker-compose-pc.yml     # PC server only ✅
├── config/config.yaml        # num_clients=1, rpi_client_ip ✅
├── client/run_client_standalone.py  # server-host=PC_IP ✅
├── client/metrics.py         # RPi temp/volts ✅
├── client/raspberrypi_metrics.py   # vcgencmd ✅
├── deployment/
│   ├── deploy_rpi.sh        # One-command deploy ✅
│   └── rpi_client_run.sh    # RPi launcher ✅
└── README.md                # This guide ✅
```

**Migration Status:** ✅ Production-ready for real RPi testing!

---

*(Original content preserved below)*

### Original Docker Setup (Multi-Container)
```
docker-compose up  # 10 simulated clients
```

### Prérequis (Original)
- Python 3.9+
- Docker Desktop (pour simulation)

## 👥 Auteur

- **SAGHOUGH Salma**
- Encadrant : Mr Yann BEN MAISSA

