# XFL-RPiLab

Framework distribué et paramétrable pour l'évaluation expérimentale du Federated Learning sur Raspberry Pi.

## 📋 Description

XFL-RPiLab est un framework de Federated Learning (FL) layer-wise conçu pour fonctionner sur un testbed de Raspberry Pi. Il permet d'évaluer expérimentalement différentes stratégies de FL dans des conditions réalistes.

## 🎯 Objectifs

- Évaluation expérimentale de stratégies FL layer-wise
- Exécution sur hardware réel (Raspberry Pi)
- Collecte automatique de métriques (système, réseau, apprentissage)
- Génération de résultats scientifiques reproductibles

## 🏗️ Architecture

```
XFL-RPiLab/
├── config/          # Configuration des expériences
├── server/          # Serveur central FL
├── client/          # Client FL (Raspberry Pi)
├── experiments/     # Orchestration des expériences
├── dashboard/       # Visualisation (optionnel)
└── logs/            # Logs et métriques
```

## 🚀 Installation & Usage (RPi Migration ✅ COMPLETE!)

### Prérequis
**PC (Server):**
- Docker Desktop
- IP accessible depuis RPi: `192.168.100.68`

**RPi (Client):**
- IP: `192.168.100.41`, user: `pi1`, pass: `1234`
- SSH passwordless (`ssh-keygen`, `ssh-copy-id pi1@192.168.100.41`)

### 1️⃣ Quick Start - Single Real RPi Client
```
# PC: Start server
docker compose -f docker-compose-pc.yml up

# PC: Deploy to RPi (auto disk cleanup + CPU torch)
bash deployment/deploy_rpi.sh

# Dashboard: http://localhost:5001 → START ROUND
```

**Troubleshooting Disk Space:**
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

### Setup

#### Option 1: Installation locale (développement)

```bash
# Cloner le repository
git clone <your-repo-url>
cd XFL-RPiLab

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Installer PostgreSQL localement ou utiliser Docker pour la base de données
# Pour Windows, installer PostgreSQL depuis https://www.postgresql.org/download/windows/
# Créer la base de données 'xfl_metrics' avec utilisateur 'postgres' et mot de passe '02069812'
```

#### Option 2: Utilisation avec Docker (recommandé)

```bash
# Cloner le repository
git clone <your-repo-url>
cd XFL-RPiLab

# Lancer l'expérimentation complète avec Docker
python run_docker_experiment.py
```

Cette commande va :
- Construire l'image Docker
- Démarrer PostgreSQL, le serveur FL, le dashboard et les clients
- Ouvrir le dashboard dans votre navigateur

#### Option 3: Serveur Docker + clients natifs Raspberry Pi

Pour déployer les clients directement sur les Raspberry Pi sans Docker :

1. Démarrez le serveur sur votre PC :

```bash
docker compose -f docker-compose-server.yml up -d
```

2. Sur chaque Raspberry Pi, exécutez le script de setup :

```bash
# Transférez les fichiers setup_pi_client.sh et run_client_pi.sh sur le Pi
scp setup_pi_client.sh pi@192.168.100.41:~/
scp run_client_pi.sh pi@192.168.100.41:~/

# Sur le Pi, exécutez le setup
ssh pi@192.168.100.41
chmod +x setup_pi_client.sh
./setup_pi_client.sh
```

3. Transférez le code et les données sur le Pi :

```bash
# Option optimisée (recommandée) - seulement ~187 MB au lieu de 2.1 GB
# Sur Windows
copy_to_pi.bat 192.168.100.41 0

# Ou manuellement :
mkdir temp_client
xcopy client temp_client\client /E /I /H /Y
xcopy config temp_client\config /E /I /H /Y
xcopy server temp_client\server /E /I /H /Y
copy requirements.txt temp_client\
xcopy data\cifar-100-python temp_client\data\cifar-100-python /E /I /H /Y
copy run_client_pi.sh temp_client\
copy setup_pi_client.sh temp_client\
scp -r temp_client pi@192.168.100.41:~/XFL-RPiLab
ssh pi@192.168.100.41 "cd ~/XFL-RPiLab && mv temp_client/* . && rm -rf temp_client"

# Option complète (plus lente)
# scp -r . pi@192.168.100.41:~/XFL-RPiLab
```

4. Lancez le client sur le Pi :

```bash
ssh pi@192.168.100.41
cd ~/XFL-RPiLab
./run_client_pi.sh 0 192.168.100.68 1 CIFAR100
```

Où :
- `0` est l'ID unique du client (0, 1, 2, etc.)
- `192.168.100.68` est l'IP de votre PC
- `1` est le nombre total de clients Pi
- `CIFAR100` est le dataset

#### Option 4: Serveur Docker + clients Docker sur Raspberry Pi

Si vous préférez utiliser Docker sur les Pi (comme dans l'ancienne version) :

1. Démarrez le serveur sur votre PC :

```bash
docker compose -f docker-compose.server.yml up -d
```

2. Déployez le client sur un Raspberry Pi (exemple pour `192.168.100.41`) :

```bash
bash deployment/deploy_simple.sh 192.168.100.41 pi raspberry 0 192.168.100.68 5000 1 CIFAR100 64 1 true
```

- `0` est l'ID client du Raspberry Pi
- `192.168.100.68` doit être l'IP de votre PC accessible depuis le Pi
- `1` est le nombre total de clients RPi
- `CIFAR100` est le dataset utilisé par le client
- `true` active la construction directe sur le Raspberry Pi (utile si la compilation ARM locale échoue)

3. Vérifiez le client sur le Pi :

```bash
ssh pi@192.168.100.41 docker ps
```

## 📊 Utilisation

*(À compléter au fur et à mesure du développement)*

## 👥 Auteur

- **SAGHOUGH Salma**
- Encadrant : Mr Yann BEN MAISSA

