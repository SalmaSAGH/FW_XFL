# FW_XFL

Framework distribué et paramétrable pour l'évaluation expérimentale du Federated Learning sur Raspberry Pi.

## 📋 Description

FW_XFL est un framework de Federated Learning (FL) layer-wise publié à l'origine par Rachid EL MOKADEM, Yann BEN MAISSA et Zineb EL AKKAOUI conçu pour fonctionner sur un testbed de Raspberry Pi. Il permet d'évaluer expérimentalement différentes stratégies de FL dans des conditions réalistes.

## 🎯 Objectifs
PFE proposé par Yann BEN MAISSA (INPT). Le but est de développer un framework qui permet de :
- Réaliser l’évaluation expérimentale de stratégies de Federated Learning (FL) layer-wise.
- Exécuter et valider les expériences sur du hardware réel, notamment des dispositifs embarqués de type Raspberry Pi.
- Assurer la collecte automatique de métriques système (CPU, mémoire, énergie), réseau (latence, bande passante) et d’apprentissage (accuracy, loss, convergence).
- Générer des résultats scientifiques reproductibles grâce à une orchestration automatisée, un versioning du code et des configurations expérimentales.

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

## 🚀 Installation

### Prérequis

- Python 3.9+
- Docker Desktop (pour simulation)

### Setup

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
```

## 📊 Utilisation

- Client actuel : Docker containers
- Pour tester la version actuelle:
  ```bash
  python run_docker_experiment.py
  ```

## 👥 Auteur

- **SAGHOUGH Salma**
- Encadrant : Mr Yann BEN MAISSA

