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

*(À compléter au fur et à mesure du développement)*

## 👥 Auteur

- **SAGHOUGH Salma**
- Encadrant : Mr Yann BEN MAISSA

