#!/bin/bash
# Script d'installation du service FL Client sur Raspberry Pi
# Ce script configure le démarrage automatique du client Federated Learning
#
# Usage: ./setup_fl_client_service.sh [OPTIONS]
#
# Options:
#   --client-id ID      ID du client (défaut: 0)
#   --server-host IP    IP du serveur (défaut: 192.168.100.68)
#   --server-port PORT  Port du serveur (défaut: 5000)
#   --dataset DATASET   Dataset (défaut: MNIST)
#   --num-clients N     Nombre de clients total (défaut: 1)
#   --data-dir PATH     Chemin des données (défaut: /home/pi1/XFL-RPiLab/data)
#   --help              Afficher cette aide
#
# Exemple:
#   ./setup_fl_client_service.sh --client-id 1 --server-host 192.168.100.68

# Valeurs par défaut
CLIENT_ID=0
SERVER_HOST="192.168.100.68"
SERVER_PORT=5000
DATASET="MNIST"
NUM_CLIENTS=1
MODE="real-hardware"
DATA_DIR="/home/pi1/XFL-RPiLab/data"
PROJECT_DIR="/home/pi1/XFL-RPiLab"

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --client-id)
            CLIENT_ID="$2"
            shift 2
            ;;
        --server-host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --client-id ID      ID du client (défaut: 0)"
            echo "  --server-host IP    IP du serveur (défaut: 192.168.100.68)"
            echo "  --server-port PORT  Port du serveur (défaut: 5000)"
            echo "  --dataset DATASET   Dataset (défaut: MNIST)"
            echo "  --num-clients N     Nombre de clients total (défaut: 1)"
            echo "  --data-dir PATH     Chemin des données"
            echo ""
            echo "Exemple:"
            echo "  $0 --client-id 1 --server-host 192.168.100.68"
            exit 0
            ;;
        *)
            echo "Option inconnue: $1"
            echo "Utilisez --help pour voir les options disponibles"
            exit 1
            ;;
    esac
done

VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
CLIENT_SCRIPT="$PROJECT_DIR/client/run_client_raspberry_pi.py"

echo "=========================================="
echo "Installation du service FL Client"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Client ID:    $CLIENT_ID"
echo "  Server Host:  $SERVER_HOST"
echo "  Server Port:  $SERVER_PORT"
echo "  Dataset:      $DATASET"
echo "  Num Clients:  $NUM_CLIENTS"
echo "  Data Dir:     $DATA_DIR"
echo ""

# Vérifier que le script existe
if [ ! -f "$CLIENT_SCRIPT" ]; then
    echo "❌ Erreur: Script client non trouvé: $CLIENT_SCRIPT"
    exit 1
fi

# Vérifier que le venv existe
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Erreur: Virtualenv non trouvé: $VENV_PYTHON"
    echo "   Créez d'abord le venv avec: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Créer le fichier de service systemd (nom unique par client)
SERVICE_NAME="fl-client-${CLIENT_ID}.service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
LOG_FILE="/home/pi1/fl-client-${CLIENT_ID}.log"

echo "📝 Création du service systemd: $SERVICE_NAME"

sudo tee $SERVICE_FILE > /dev/null << 'EOF'
[Unit]
Description=Federated Learning Client for XFL-RPiLab
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=__PROJECT_DIR__
ExecStart=__VENV_PYTHON__ __CLIENT_SCRIPT__ --client-id __CLIENT_ID__ --server-host __SERVER_HOST__ --server-port __SERVER_PORT__ --dataset __DATASET__ --num-clients __NUM_CLIENTS__ --mode __MODE__ --data-dir __DATA_DIR__
Restart=always
RestartSec=10
StandardOutput=append:__LOG_FILE__
StandardError=append:__LOG_FILE__

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Remplacer les variables dans le fichier
sudo sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" $SERVICE_FILE
sudo sed -i "s|__VENV_PYTHON__|$VENV_PYTHON|g" $SERVICE_FILE
sudo sed -i "s|__CLIENT_SCRIPT__|$CLIENT_SCRIPT|g" $SERVICE_FILE
sudo sed -i "s|__CLIENT_ID__|$CLIENT_ID|g" $SERVICE_FILE
sudo sed -i "s|__SERVER_HOST__|$SERVER_HOST|g" $SERVICE_FILE
sudo sed -i "s|__SERVER_PORT__|$SERVER_PORT|g" $SERVICE_FILE
sudo sed -i "s|__DATASET__|$DATASET|g" $SERVICE_FILE
sudo sed -i "s|__NUM_CLIENTS__|$NUM_CLIENTS|g" $SERVICE_FILE
sudo sed -i "s|__MODE__|$MODE|g" $SERVICE_FILE
sudo sed -i "s|__DATA_DIR__|$DATA_DIR|g" $SERVICE_FILE
sudo sed -i "s|__LOG_FILE__|$LOG_FILE|g" $SERVICE_FILE

# Recharger systemd
echo "🔄 Rechargement de systemd..."
sudo systemctl daemon-reload

# Activer le service
echo "✅ Activation du service au démarrage..."
sudo systemctl enable $SERVICE_NAME

# Démarrer le service
echo "🚀 Démarrage du service..."
sudo systemctl start $SERVICE_NAME

# Vérifier le statut
echo ""
echo "=========================================="
echo "Vérification du statut du service"
echo "=========================================="
sudo systemctl status $SERVICE_NAME --no-pager

echo ""
echo "=========================================="
echo "Installation terminée!"
echo "=========================================="
echo ""
echo "Commandes utiles:"
echo "  - Voir les logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  - Redémarrer:       sudo systemctl restart $SERVICE_NAME"
echo "  - Arrêter:          sudo systemctl stop $SERVICE_NAME"
echo "  - Statut:           sudo systemctl status $SERVICE_NAME"
echo ""
echo "Le client démarrera automatiquement après chaque reboot."