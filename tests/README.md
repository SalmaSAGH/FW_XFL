# Tests du projet XFL-RPiLab

Ce dossier contient les tests automatisés pour le projet de fin d'étude. Ils couvrent trois axes importants :

1. Tests unitaires
2. Tests d'intégration
3. Tests d'API serveur

## Comment exécuter les tests

Dans la racine du projet :

```powershell
.\venv\Scripts\Activate.ps1
pytest -q
```

## Fichiers de test et leur rôle

- `tests/test_config_parser.py`
  - Vérifie la lecture et la validation des fichiers de configuration YAML.
  - Teste les règles métiers comme `clients_per_round <= num_clients`.

- `tests/test_dataset_partition.py`
  - Vérifie la partition des données en mode IID et non-IID.
  - Teste la création de sous-ensembles par client.

- `tests/test_client_model.py`
  - Vérifie la création de modèles de réseau de neurones.
  - Teste le passage avant (`forward pass`) sur des entrées factices.

- `tests/test_server_helpers.py`
  - Vérifie la sérialisation et la désérialisation des poids du modèle.

- `tests/test_flask_api.py`
  - Vérifie les endpoints Flask essentiels.
  - Utilise des mocks pour isoler la logique serveur sans base de données réelle.

- `tests/test_integration_data_loader.py`
  - Vérifie que le loader de données client fonctionne bien avec un patch du chargement de dataset.

## Pourquoi ces tests sont utiles pour un rapport de PFE

- Ils montrent que ton projet est testé à plusieurs niveaux : logique métier, machine learning et API.
- Ils illustrent le concept de *tests unitaires* et de *tests d'intégration*.
- Ils permettent de documenter la robustesse du système et la qualité du code.

## Points à expliquer dans le rapport

- La différence entre tests unitaires et tests d'intégration.
- Le rôle du `pytest.ini` pour centraliser la configuration de test.
- L'utilisation de `monkeypatch` pour simuler des composants externes.
- L'intérêt d'avoir une suite exécutable avec `pytest -q`.
