# FW_XFL

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A distributed and configurable framework for the experimental evaluation of **Layer-wise Federated Learning (FL)** on Raspberry Pi devices.

---

## 📑 Table of Contents

- [Description](#-description)
- [Objectives](#-objectives)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Metrics Collected](#-metrics-collected)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [Author](#-author)

---

## 📋 Description

**FW_XFL** is a Layer-wise Federated Learning (FL) framework originally published by  
Rachid EL MOKADEM, Yann BEN MAISSA, and Zineb EL AKKAOUI.

It is designed to operate on a **Raspberry Pi testbed**, enabling realistic and reproducible experimental evaluation of Federated Learning strategies under real-world hardware constraints.

The framework supports both:
- 🧪 Real hardware experimentation (Raspberry Pi cluster)
- 🐳 Docker-based simulation for scalability testing

---

## 🎯 Objectives

This project is part of a Final Year Project (PFE) proposed by **Yann BEN MAISSA (INPT)**.

The framework aims to:

- Perform experimental evaluation of Layer-wise Federated Learning strategies  
- Execute and validate experiments on real embedded devices (Raspberry Pi)  
- Automatically collect:
  - System metrics (CPU, memory, energy)
  - Network metrics (latency, bandwidth)
  - Learning metrics (accuracy, loss, convergence)
- Ensure scientific reproducibility through:
  - Automated orchestration
  - Code versioning
  - Experiment configuration management

---

## 🏗️ Architecture

```
XFL-RPiLab/
├── config/ # Experiment configuration
├── server/ # Central FL server
├── client/ # FL client (Raspberry Pi)
├── experiments/ # Experiment orchestration
├── dashboard/ # Visualization (optional)
└── logs/ # Logs and metrics
```


## 🚀 Installation

### Prerequisites

- Python 3.9+
- Docker Desktop (for simulation)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd XFL-RPiLab

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

📊 Usage
Current client implementation: Docker containers
🐳 Docker Simulation Mode
To test the current version:
  ```bash
  docker-compose up
  ```

👥 Author

- SAGHOUGH Salma
- Supervisor: Mr. Yann BEN MAISSA
