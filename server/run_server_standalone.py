"""
Standalone server script for Docker deployment
"""

import sys
import os
import time
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import create_server
from client import create_model, create_dataloaders
from client.model import DATASET_CONFIG  # ← source unique de vérité pour dataset→model

# NEW: Config parsing for dynamic dataset init
try:
    from config.config_parser import load_config
    config_cifar100 = load_config("config/config_cifar100.yaml")
    initial_dataset = config_cifar100.dataset.name  # CIFAR100
    initial_dist = "iid"
    print(f"✅ Loaded config_cifar100.yaml: dataset={initial_dataset}")
except Exception as e1:
    try:
        config_main = load_config("config/config.yaml")
        initial_dataset = config_main.dataset.name  # CIFAR100 fallback
        initial_dist = "iid"
        print(f"✅ Loaded config.yaml: dataset={initial_dataset}")
    except Exception as e2:
        # Defaults fallback
        initial_dataset = 'CIFAR100'
        initial_dist = 'iid'
        print(f"✅ Using defaults: dataset={initial_dataset}, dist={initial_dist} (errors: {e1}, {e2})")

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from server.server import app, fl_config

import flask
import threading



def wait_for_postgres(host="postgres", port=5432, user="postgres",
                      password="newpassword", db="xfl_metrics", timeout=60):
    """Wait for PostgreSQL to be ready"""
    print(f"⏳ Waiting for PostgreSQL at {host}:{port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            conn = psycopg2.connect(host=host, port=port, user=user,
                                    password=password, dbname=db)
            conn.close()
            print("✅ PostgreSQL is ready!")
            return True
        except psycopg2.OperationalError:
            print("   PostgreSQL not ready, waiting...")
            time.sleep(2)
    print("❌ Timeout waiting for PostgreSQL")
    return False


def _get_model_and_loader(dataset_name: str, distribution: str = 'iid',
                          data_dir: str = '/app/data'):
    """
    Crée le bon modèle ET le bon test_loader pour un dataset donné.
    Utilise DATASET_CONFIG comme source unique de vérité.
    Retourne (model, test_loader) ou lève une exception.
    """
    cfg = DATASET_CONFIG.get(dataset_name)
    if cfg is None:
        raise ValueError(f"Dataset inconnu: {dataset_name}. "
                         f"Choix disponibles: {list(DATASET_CONFIG.keys())}")

    model_name, num_classes, in_channels, input_size = cfg

    print(f"  📐 dataset={dataset_name} | model={model_name} | "
          f"num_classes={num_classes} | in_channels={in_channels} | "
          f"input_size={input_size}")

    # Crée le modèle avec les BONS paramètres
    model = create_model(
        model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size
    )

    # Get num_clients from config or environment (default 5 for Docker, 1 for physical)
    num_clients = int(os.getenv('NUM_CLIENTS', '5'))
    
    # Charge le test_loader (utilise num_clients pour la partition)
    _, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        num_clients=num_clients,
        batch_size=256,
        distribution=distribution,
        data_dir=data_dir,
        seed=42
    )

    return model, test_loader


def reload_server_model_and_data(dataset_name: str, distribution: str = 'iid',
                                  data_dir: str = '/app/data'):
    """
    Recrée le global_model et le test_loader du serveur quand le dataset change.
    Appelé depuis le patch de /api/config/save en arrière-plan.
    """
    import server.server as server_module

    if server_module.fl_server is None:
        print("⚠️  fl_server not yet initialized, skipping reload")
        return

    print(f"\n🔄 Server reloading model+data: dataset={dataset_name}, "
          f"distribution={distribution}")

    try:
        new_model, new_test_loader = _get_model_and_loader(
            dataset_name, distribution, data_dir
        )
    except Exception as e:
        print(f"❌ reload_server_model_and_data failed: {e}")
        return

    # Mise à jour thread-safe du serveur FL
    with server_module.fl_server.lock:
        server_module.fl_server.global_model = new_model
        server_module.fl_server.test_loader  = new_test_loader

    print(f"✅ Server model+data reloaded for dataset={dataset_name}")


def main():
    if not wait_for_postgres():
        print("❌ Cannot connect to PostgreSQL, exiting...")
        sys.exit(1)

    num_clients = int(os.getenv('NUM_CLIENTS', '5'))
    db_url = (os.getenv('DB_URL') or
              os.getenv('DATABASE_URL',
                        'postgresql://postgres:newpassword@postgres:5432/xfl_metrics'))

    print("=" * 70)
    print("XFL-RPiLab FL Server (Docker)")
    print("=" * 70)
    print(f"Starting server on 0.0.0.0:5000")
    print(f"Expected clients : {num_clients}")
    print(f"Database         : {db_url}")
    print("=" * 70 + "\n")

# ── Initial dataset from config/DB (FIXED) ────────────────────────────────
    print(f"📊 Initial dataset from config: {initial_dataset} ({initial_dist})")

    try:
        model, test_loader = _get_model_and_loader(
            initial_dataset, initial_dist, '/app/data'
        )

    except Exception as e:
        print(f"❌ Failed to initialize model/data: {e}")
        sys.exit(1)

    print("\n📊 Creating FL server...")
    server = create_server(
        global_model=model,
        test_loader=test_loader,
        aggregation_strategy="fedavg",
        num_rounds=100,
        clients_per_round=num_clients,
        db_url=db_url
    )
    
    # ── Initialize physical client manager and routes ───────────────────────
    # This allows the server to detect and use registered Raspberry Pi clients
    try:
        from server.physical_client_manager import create_physical_client_manager
        physical_mgr = create_physical_client_manager("http://localhost:5000")
        print("✅ Physical client manager initialized")
        
        # Register physical client routes directly with Flask app
        from server.server import init_physical_client_routes
        init_physical_client_routes(physical_mgr)
        print("✅ Physical client routes registered")
    except Exception as e:
        print(f"⚠️ Could not initialize physical client manager: {e}")

    # ── Monkey-patch /api/config/save ────────────────────────────────────────
    # Déclenche reload_server_model_and_data en arrière-plan si le dataset change.
    import server.server as server_module
    _original_save_config = server_module.save_config

    def _patched_save_config():
        # 1. Exécuter le handler original (met à jour fl_config)
        response = _original_save_config()

        # 2. Lire le nouveau dataset/distribution depuis fl_config
        new_dataset = server_module.fl_config.get('dataset', 'MNIST')
        new_dist    = server_module.fl_config.get('dataDistribution', 'iid')

        print(f"\n🔔 Config saved: dataset={new_dataset}, distribution={new_dist}")

        # 3. Recharger en arrière-plan pour ne pas bloquer la réponse HTTP
        t = threading.Thread(
            target=reload_server_model_and_data,
            args=(new_dataset, new_dist, '/app/data'),
            daemon=True
        )
        t.start()

        return response

    app.view_functions['save_config'] = _patched_save_config

    print("\n⚠️  Server is ready but NO round started.")
    print("   Use the dashboard or API to start rounds manually.")
    print("\n✅ Server initialization complete!")
    print("🚀 Starting Flask server...\n")

    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True,
            processes=1, use_reloader=False)


if __name__ == "__main__":
    main()