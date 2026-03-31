"""
Flask server for Federated Learning with XFL support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.utils.data import DataLoader
import pickle
import base64
import hashlib
import secrets
import os
import uuid                          # ← NEW: for session_id generation
import socket                        # ← NEW: for hostname in server_sessions
import psycopg2
import psycopg2.extras
import psycopg2.pool
from collections import OrderedDict
from typing import Dict, List, Any
import time
import threading
import signal
import sys

from .strategy import create_aggregation_strategy, XFL, FedAvg
from .metrics import ServerMetricsCollector
from .dse import start_dse_sweep, list_dse_sessions, load_dse_session, get_dse_job_status
from client.model import create_model, DATASET_CONFIG  # ← DYNAMIC MODEL SUPPORT
from client.dataset import load_dataset  # ← TEST DATA LOADER
import gc


# Database URL for dashboard APIs
DB_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:newpassword@localhost:5432/xfl_metrics')

# ── NEW: Generate a unique session_id once per process lifetime ───────────────
# This UUID is created when the server starts (docker-compose up) and never
# changes until the process exits (docker-compose down).  Every FL round
# started in this process will be tagged with this session_id so that the
# History page can group rounds into true "sessions" instead of relying on
# round-number gap detection.
CURRENT_SESSION_ID = str(uuid.uuid4())

# Connection pool for database connections
_connection_pool = None
_pool_lock = threading.Lock()


def _init_connection_pool():
    """Initialize the connection pool for database connections"""
    global _connection_pool
    if _connection_pool is not None:
        return _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            return _connection_pool
            
        try:
            db_url = DB_URL.replace('postgresql://', '')
            if '@' in db_url:
                auth, host_db = db_url.split('@')
                user, password = auth.split(':')
                if '/' in host_db:
                    host_port, dbname = host_db.split('/')
                    if ':' in host_port:
                        host, port = host_port.split(':')
                    else:
                        host = host_port
                        port = '5432'
                else:
                    host = host_db
                    port = '5432'
                    dbname = 'postgres'
            else:
                user = 'postgres'
                password = 'password'
                host = 'localhost'
                port = '5432'
                dbname = 'postgres'
            
            _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=5,
                connect_timeout=5,
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname
            )
            print("Server connection pool initialized (min: 2, max: 10)")
            return _connection_pool
        except Exception as e:
            print(f"WARNING: Warning: Could not initialize connection pool: {e}")
            return None


def _get_connection():
    """Get a connection from the pool"""
    global _connection_pool
    
    if _connection_pool is None:
        _init_connection_pool()
    
    if _connection_pool:
        try:
            conn = _connection_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except:
                _connection_pool.putconn(conn, close=True)
                conn = _connection_pool.getconn()
                return conn
        except Exception as e:
            print(f"WARNING: Warning: Could not get connection from pool: {e}")
    
    return psycopg2.connect(DB_URL)


def _return_connection(conn):
    """Return a connection to the pool"""
    global _connection_pool
    
    if _connection_pool and conn:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            print(f"WARNING: Warning: Could not return connection to pool: {e}")
            try:
                conn.close()
            except:
                pass
    elif conn:
        try:
            conn.close()
        except:
            pass


# ── NEW: Register the current session in the database ────────────────────────
def _register_server_session():
    """
    Insert one row into server_sessions for the current docker-compose up.
    Called once at startup, after the DB tables have been initialised.
    """
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        hostname = socket.gethostname()
        cursor.execute("""
            INSERT INTO server_sessions (session_id, started_at, hostname)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
        """, (CURRENT_SESSION_ID, time.time(), hostname))
        conn.commit()
        conn.close()
        print(f"Server session registered: {CURRENT_SESSION_ID} (host: {hostname})")
    except Exception as e:
        print(f"WARNING: Warning: Could not register server session: {e}")


def init_fl_config_database():
    """Initialize FL config table in the database"""
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fl_config (
            id SERIAL PRIMARY KEY,
            config_key VARCHAR(255) UNIQUE NOT NULL,
            config_value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("FL config table initialized")


def save_fl_config_value(key: str, value: str):
    """Save a configuration value to the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fl_config (config_key, config_value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (config_key) 
            DO UPDATE SET config_value = EXCLUDED.config_value, updated_at = EXCLUDED.updated_at
        """, (key, value, time.time()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving config {key}: {e}")
        return False


def get_fl_config_value(key: str, default: str = None) -> str:
    """Get a configuration value from the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT config_value FROM fl_config WHERE config_key = %s", (key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        return default
    except Exception as e:
        print(f"Error getting config {key}: {e}")
        return default


def save_fl_config_to_db(config: dict):
    """Save the entire FL configuration to the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        for key, value in config.items():
            cursor.execute("""
                INSERT INTO fl_config (config_key, config_value, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (config_key) 
                DO UPDATE SET config_value = EXCLUDED.config_value, updated_at = EXCLUDED.updated_at
            """, (key, str(value), time.time()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving FL config: {e}")
        return False


def load_fl_config_from_db() -> dict:
    """Load the FL configuration from the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT config_key, config_value FROM fl_config")
        results = cursor.fetchall()
        conn.close()
        
        config = {}
        for key, value in results:
            if value.isdigit():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except:
                    config[key] = value
        
        return config
    except Exception as e:
        print(f"Error loading FL config: {e}")
        return {}


def init_server_state_database():
    """Initialize FL server state table in the database"""
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fl_server_state (
            id SERIAL PRIMARY KEY,
            state_key VARCHAR(255) UNIQUE NOT NULL,
            state_value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("FL server state table initialized")


def save_server_state(key: str, value: str):
    """Save server state to the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fl_server_state (state_key, state_value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (state_key) 
            DO UPDATE SET state_value = EXCLUDED.state_value, updated_at = EXCLUDED.updated_at
        """, (key, str(value), time.time()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving server state {key}: {e}")
        return False


def get_server_state(key: str, default: str = None) -> str:
    """Get server state from the database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT state_value FROM fl_server_state WHERE state_key = %s", (key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        return default
    except Exception as e:
        print(f"Error getting server state {key}: {e}")
        return default


def init_user_database():
    """Initialize users and sessions tables in the database"""
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            token VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(255) NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print("Users and sessions tables initialized")


def get_user_from_db(username: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT username, password_hash, created_at FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {'username': result[0], 'password': result[1], 'created_at': result[2]}
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None


def create_user_in_db(username: str, password_hash: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (%s, %s, %s)",
            (username, password_hash, time.time())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False


def create_session_in_db(token: str, username: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (token, username, created_at) VALUES (%s, %s, %s)",
            (token, username, time.time())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating session: {e}")
        return False


def get_session_from_db(token: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT token, username, created_at FROM sessions WHERE token = %s", (token,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {'token': result[0], 'username': result[1], 'created_at': result[2]}
        return None
    except Exception as e:
        print(f"Error getting session: {e}")
        return None


def delete_session_from_db(token: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = %s", (token,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting session: {e}")
        return False


class FLServer:
    """
    Federated Learning Server with XFL support
    """
    
    def __init__(
        self,
        global_model: torch.nn.Module,
        aggregation_strategy: str = "fedavg",
        num_rounds: int = 10,
        clients_per_round: int = 5,
        db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics",
        xfl_strategy: str = "all_layers",
        xfl_param: int = 3
    ):
        self.global_model = global_model
        self.current_dataset_name = "MNIST"  # ← TRACK CURRENT DATASET
        self.aggregation_strategy = create_aggregation_strategy(
            aggregation_strategy,
            xfl_strategy=xfl_strategy,
            xfl_param=xfl_param
        )
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        
        self.metrics_collector = ServerMetricsCollector(db_url)
        
        # ── NEW: attach the process-level session_id to every server instance ─
        self.session_id = CURRENT_SESSION_ID

        self.current_round = self._load_current_round()
        self.round_in_progress = False
        self.client_submissions = []
        self.selected_clients = []
        self.lock = threading.Lock()
        
        self.test_loader = None
        
        print(f"FLServer initialized")
        print(f"   Strategy: {self.aggregation_strategy.name}")
        print(f"   XFL: {xfl_strategy}")
        print(f"   Total rounds: {self.num_rounds}")
        print(f"   Clients per round: {self.clients_per_round}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Current round (loaded from DB): {self.current_round}")
    
    def _load_current_round(self) -> int:
        """
        Load current round from database.
        ── CHANGED: only look at rounds that belong to the CURRENT session_id ──
        This guarantees that after docker-compose down/up, round numbering
        restarts from 0 within the new session, rather than continuing from the
        previous session's last round.
        """
        try:
            conn = psycopg2.connect(DB_URL)
            cursor = conn.cursor()

            # Get the highest completed round for THIS session only
            cursor.execute("""
                SELECT MAX(round_number)
                FROM round_metrics
                WHERE session_id = %s AND global_test_accuracy IS NOT NULL
            """, (self.session_id,))
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return result[0]

        except Exception as e:
            print(f"WARNING:  Warning: Could not load current round from database: {e}")
        return 0
    
    def _save_current_round(self):
        """Save current round to database (kept for compatibility)"""
        try:
            save_server_state('current_round', str(self.current_round))
        except Exception as e:
            print(f"WARNING:  Warning: Could not save current round to database: {e}")
    
    def start_round(self) -> Dict[str, Any]:
        """Start a new FL round"""
        with self.lock:
            if self.current_round >= self.num_rounds:
                return {
                    "status": "completed",
                    "message": "All rounds completed"
                }

            if self.round_in_progress:
                return {
                    "status": "in_progress",
                    "round": self.current_round,
                    "message": "Round already in progress"
                }

            # Reload config from database to get latest values
            try:
                latest_config = load_fl_config_from_db()
                if latest_config:
                    for key, value in latest_config.items():
                        fl_config[key] = value
                    
                    if 'clientsPerRound' in latest_config:
                        self.clients_per_round = int(latest_config['clientsPerRound'])
                        print(f"Using clientsPerRound from config: {self.clients_per_round}")
                    
                    if 'numRounds' in latest_config:
                        self.num_rounds = int(latest_config['numRounds'])
                        print(f"Using numRounds from config: {self.num_rounds}")
            except Exception as e:
                print(f"WARNING:  Warning: Could not reload config from database: {e}")

            self.current_round += 1
            self.round_in_progress = True
            self.client_submissions = []

            self._save_current_round()

            # Save placeholder round metrics tagged with session_id
            try:
                self.metrics_collector.store_round_metrics(
                    round_number=self.current_round,
                    num_clients=0,
                    aggregation_time=0,
                    global_test_loss=None,
                    global_test_accuracy=None,
                    total_samples=0,
                    session_id=self.session_id,          # ← pass session_id
                    additional_metrics={
                        "status": "in_progress",
                        "clients_expected": self.clients_per_round
                    }
                )
                print(f"Round {self.current_round} metrics placeholder saved (session: {self.session_id})")
            except Exception as e:
                print(f"WARNING:  Warning: Could not save round metrics placeholder: {e}")

            import random
            total_clients = int(fl_config.get('numClients', 40))
            actual_clients_per_round = min(self.clients_per_round, total_clients)
            self.selected_clients = random.sample(range(total_clients), actual_clients_per_round)

            xfl_info = self.aggregation_strategy.get_xfl_info()
            print(f"\n🔄 Starting Round {self.current_round}/{self.num_rounds}")
            print(f"   Session: {self.session_id}")
            print(f"   Total clients available: {total_clients}")
            print(f"   Clients selected per round: {actual_clients_per_round}")
            print(f"   Selected clients: {self.selected_clients}")
            print(f"   XFL Strategy: {xfl_info['strategy']}")

            return {
                "status": "started",
                "round": self.current_round,
                "clients_expected": actual_clients_per_round,
                "selected_clients": self.selected_clients,
                "xfl_strategy": xfl_info['strategy']
            }
    
    def submit_client_update(
        self,
        client_id: int,
        model_weights: OrderedDict,
        num_samples: int,
        client_metrics: Dict[str, Any],
        quantization_meta: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """Receive model update from a client"""
        with self.lock:
            if not self.round_in_progress:
                return {
                    "status": "error",
                    "message": "No round in progress"
                }

            if any(sub["client_id"] == client_id for sub in self.client_submissions):
                return {
                    "status": "duplicate",
                    "message": f"Client {client_id} has already submitted for this round"
                }

            self.client_submissions.append({
                "client_id": client_id,
                "weights": model_weights,
                "num_samples": num_samples,
                "metrics": client_metrics,
                "quantization_meta": quantization_meta
            })

            self.metrics_collector.store_client_metrics(
                self.current_round,
                client_id,
                client_metrics
            )

            print(f"   Received update from Client {client_id} "
                  f"({len(self.client_submissions)}/{self.clients_per_round})")

            if len(self.client_submissions) >= self.clients_per_round:
                self._aggregate_round()

            return {
                "status": "received",
                "round": self.current_round,
                "submissions": len(self.client_submissions)
            }
    
    def _aggregate_round(self):
        """Aggregate client updates using XFL strategy"""
        xfl_info = self.aggregation_strategy.get_xfl_info()
        print(f"\nAggregating {len(self.client_submissions)} clients with XFL: {xfl_info['strategy']}")

        start_time = time.time()

        try:
            client_weights = [sub["weights"] for sub in self.client_submissions]
            client_num_samples = [sub["num_samples"] for sub in self.client_submissions]

            has_quantization = any(sub.get("quantization_meta") for sub in self.client_submissions)

            if has_quantization and xfl_info['variant'] == 'quantization':
                print("   🔧 Applying dequantization before aggregation...")
                client_weights = self._dequantize_client_weights(client_weights)

            aggregated_weights = self.aggregation_strategy.aggregate(
                client_weights,
                client_num_samples
            )

            # FIXED: Ensure all tensors are float32 before model update
            def ensure_float32(weights):
                for name, tensor in weights.items():
                    if tensor.dtype != torch.float32:
                        print(f"🔧 Converting '{name}': {tensor.dtype} → float32 "
                              f"(shape: {tensor.shape})")
                        weights[name] = tensor.float()
                return weights
            
            aggregated_weights = ensure_float32(aggregated_weights)

            # Apply model update with robust error handling
            model_updated = False
            try:
                if isinstance(self.aggregation_strategy, XFL):
                    current_state = self.global_model.state_dict()
                    current_state.update(aggregated_weights)
                    self.global_model.load_state_dict(current_state, strict=False)
                else:
                    self.global_model.load_state_dict(aggregated_weights, strict=False)
                model_updated = True
                print("Global model updated successfully")
            except Exception as model_error:
                print(f"WARNING:  Model load_state_dict failed (continuing): {model_error}")
                print("   Tensors will be skipped but round metrics still saved")

            aggregation_time = time.time() - start_time

            test_loss, test_accuracy = None, None
            if self.test_loader is not None:
                try:
                    test_loss, test_accuracy = self._evaluate_global_model()
                except Exception as e:
                    print(f"WARNING:  Global model evaluation failed: {e}")
                    test_loss, test_accuracy = None, None

            xfl_info = self.aggregation_strategy.get_xfl_info()
            # ── CHANGED: pass session_id when storing the completed round ─────
            self.metrics_collector.store_round_metrics(
                round_number=self.current_round,
                num_clients=len(self.client_submissions),
                aggregation_time=aggregation_time,
                global_test_loss=test_loss,
                global_test_accuracy=test_accuracy,
                total_samples=sum(client_num_samples),
                strategy=xfl_info['strategy'],
                session_id=self.session_id,
                additional_metrics={
                    "submitted_clients": [sub["client_id"] for sub in self.client_submissions]
                    }
                    )

            print(f"Round {self.current_round} completed in {aggregation_time:.2f}s")
            print(f"   🌍 Global Test → Loss: {test_loss:.4f} | Acc: {test_accuracy:.2f}%" if test_accuracy is not None else "   🌍 Global Test → SKIPPED")

        except Exception as e:
            aggregation_time = time.time() - start_time
            print(f"ERROR: Aggregation failed for round {self.current_round}: {e}")
            print(f"   Round will be completed anyway to prevent blocking")

        self._save_current_round()

        self.round_in_progress = False
        self.client_submissions = []
        
        gc.collect()
        print(f"🧹 Garbage collection triggered after round {self.current_round}")
    
    def _evaluate_global_model(self):
        """Evaluate global model on test set"""
        self.global_model.eval()
        device = next(self.global_model.parameters()).device
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def _dequantize_client_weights(self, client_weights: List[OrderedDict]) -> List[OrderedDict]:
        dequantized_weights = []

        for i, client_weight in enumerate(client_weights):
            client_meta = self.client_submissions[i].get("quantization_meta")

            if client_meta is None:
                dequantized_weights.append(client_weight)
                continue

            dequantized = OrderedDict()
            for param_name, param in client_weight.items():
                if param_name in client_meta and client_meta[param_name]['quantized']:
                    meta = client_meta[param_name]
                    dequantized_param = param.float() * meta['scale']
                    dequantized[param_name] = dequantized_param
                else:
                    dequantized[param_name] = param.float()

            dequantized_weights.append(dequantized)

        return dequantized_weights

    def get_global_model(self) -> OrderedDict:
        with self.lock:
            weights = self.global_model.state_dict()
            # 🔍 DEBUG: Log top layer shapes for CIFAR100 troubleshooting
            debug_layers = ['conv1.weight', 'conv2.weight', 'fc1.weight']
            for layer in debug_layers:
                if layer in weights:
                    print(f"SERVER DEBUG: {layer} shape={weights[layer].shape}")
                else:
                    print(f"SERVER DEBUG: {layer} MISSING")
            print(f"SERVER DEBUG: Total params sent={len(weights)}")
            return weights
    
    def recreate_global_model(self, dataset_name: str):
        """🔧 Recreate global model when dataset changes (EMNIST fix)"""
        try:
            cfg = DATASET_CONFIG.get(dataset_name, ('SimpleCNN', 10, 1, 28))
            model_name, num_classes, in_channels, input_size = cfg
            
            old_params = sum(p.numel() for p in self.global_model.parameters())
            print(f"🔄 Recreating model: {self.current_dataset_name} ({old_params:,} params)")
            print(f"   → {dataset_name}: {model_name}, {num_classes}cls, {in_channels}ch")
            
            self.global_model = create_model(model_name, num_classes, in_channels, input_size)
            
            new_params = sum(p.numel() for p in self.global_model.parameters())
            print(f"New model created: {new_params:,} params | Dataset: {dataset_name}")
            
            self.current_dataset_name = dataset_name
            
            # Recreate test_loader
            self.test_loader = DataLoader(
                load_dataset(dataset_name, data_dir="./data", train=False),
                batch_size=256, shuffle=False, num_workers=0
            )
            print(f"Test loader updated for {dataset_name}")
            
        except Exception as e:
            print(f"ERROR: Model recreation failed: {e}")
    
    def get_server_status(self) -> Dict[str, Any]:
        with self.lock:
            xfl_info = self.aggregation_strategy.get_xfl_info()
            layer_names = list(self.global_model.state_dict().keys())
            num_layers = len(set(name.split('.')[0] for name in layer_names))
            
            total_clients = int(fl_config.get('numClients', 40))
            latest_accuracy = None
            
            try:
                conn = psycopg2.connect(DB_URL)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT global_test_accuracy
                    FROM round_metrics
                    WHERE global_test_accuracy IS NOT NULL
                    ORDER BY round_number DESC
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result and result[0] is not None:
                    latest_accuracy = result[0]
                
                conn.close()
            except Exception as e:
                print(f"WARNING: Warning: Could not get additional status from database: {e}")
            
            return {
                "current_round": self.current_round,
                "total_rounds": self.num_rounds,
                "round_in_progress": self.round_in_progress,
                "submissions_received": len(self.client_submissions),
                "clients_expected": self.clients_per_round,
                "total_clients": total_clients,
                "latest_accuracy": latest_accuracy,
                "current_dataset": self.current_dataset_name,  # ← NEW
                "submitted_clients": [sub["client_id"] for sub in self.client_submissions],
                "selected_clients": self.selected_clients if self.round_in_progress else [],
                "xfl_strategy": xfl_info['strategy'],
                "xfl_param": xfl_info['param'],
                "num_layers": num_layers,
                "session_id": self.session_id        # ← expose for debugging
            }
    
    def set_xfl_strategy(self, strategy: str, param: int = 3) -> Dict[str, Any]:
        with self.lock:
            if self.round_in_progress:
                return {
                    "status": "error",
                    "message": "Cannot change strategy during active round"
                }

            if strategy.startswith("xfl"):
                self.aggregation_strategy = create_aggregation_strategy("xfl", strategy, param)
            else:
                self.aggregation_strategy = FedAvg(xfl_strategy=strategy, xfl_param=param)

            return {
                "status": "success",
                "message": f"XFL strategy updated to {strategy}",
                "xfl_strategy": strategy,
                "xfl_param": param
            }


# Flask app
app = Flask(__name__)
CORS(app)

app.secret_key = 'xfl-rpilab-secret-key-change-in-production'

fl_server: FLServer = None

fl_config = {
    "numClients": 40,
    "numRounds": 50,
    "clientsPerRound": 5,
    "dataset": "MNIST",
    "dataDistribution": "iid",

    "strategy": "all_layers",
    "xflParam": 3,
    "localEpochs": 2,
    "batchSize": 512,
    "learningRate": 0.01,
    "networkLatency": 0,
    "networkBandwidth": 10,
    "networkPacketLoss": 0,
    "cpuLimit": 100,
    "ramLimit": 2048,
}

# Initialize database at startup
try:
    init_user_database()
    init_fl_config_database()
    init_server_state_database()
    # ── NEW: register this docker-compose up as a new session ────────────────
    _register_server_session()
except Exception as e:
    print(f"WARNING:  Database initialization warning: {e}")
    print("   Server will continue but auth features may not work properly")

# Load FL config from database at startup
try:
    loaded_config = load_fl_config_from_db()
    if loaded_config:
        fl_config.update(loaded_config)
        print("FL config loaded from database")
except Exception as e:
    print(f"WARNING:  Warning: Could not load FL config from database: {e}")


users_db = {}
sessions = {}


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save FL configuration"""
    global fl_config
    data = request.get_json()
    
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    fl_config.update(data)
    save_fl_config_to_db(fl_config)
    
    if fl_server is not None and 'strategy' in data:
        strategy = data.get('strategy', 'all_layers')
        param = data.get('xflParam', 3)
        fl_server.set_xfl_strategy(strategy, param)
    
    if fl_server is not None and 'clientsPerRound' in data:
        fl_server.clients_per_round = int(data['clientsPerRound'])
        print(f"Updated clients_per_round to {fl_server.clients_per_round}")
    
    if 'clientsPerRound' in data:
        fl_config['clientsPerRound'] = int(data['clientsPerRound'])
        print(f"Updated fl_config clientsPerRound to {fl_config['clientsPerRound']}")
    
    if fl_server is not None and 'numRounds' in data:
        old_num_rounds = fl_server.num_rounds
        fl_server.num_rounds = int(data['numRounds'])
        fl_config['numRounds'] = int(data['numRounds'])
        print(f"Updated numRounds from {old_num_rounds} to {fl_server.num_rounds}")
    
    # ← FIXED EMNIST: Recreate global_model when dataset changes
    if 'dataset' in data:
        new_dataset = data['dataset']
        if fl_server and new_dataset != fl_server.current_dataset_name:
            print(f"🔄 Server: Dataset changed {fl_server.current_dataset_name} → {new_dataset}")
            fl_server.recreate_global_model(new_dataset)
    
    return jsonify({"status": "success", "message": "Configuration saved", "config": fl_config})


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({"config": fl_config})


@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
        
        existing_user = get_user_from_db(username)
        if existing_user:
            return jsonify({"error": "Username already exists"}), 409
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        success = create_user_in_db(username, hashed_password)
        if not success:
            return jsonify({"error": "Registration failed. Please try again."}), 500
        
        return jsonify({"status": "success", "message": "User registered successfully"}), 201
    except Exception as e:
        print(f"Error in register: {e}")
        return jsonify({"error": "Registration failed. Please try again."}), 500


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    user = get_user_from_db(username)
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if user['password'] != hashed_password:
        return jsonify({"error": "Invalid username or password"}), 401
    
    token = secrets.token_hex(16)
    create_session_in_db(token, username)
    
    return jsonify({
        "status": "success", 
        "message": "Login successful",
        "token": token,
        "username": username
    }), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    data = request.get_json()
    token = data.get('token')
    
    if token:
        delete_session_from_db(token)
    
    return jsonify({"status": "success", "message": "Logged out successfully"}), 200


@app.route('/api/verify_token', methods=['POST'])
def verify_token():
    data = request.get_json()
    token = data.get('token')
    
    session = get_session_from_db(token)
    if session:
        return jsonify({"status": "valid", "username": session['username']}), 200
    
    return jsonify({"status": "invalid"}), 401


def weights_to_base64(weights: OrderedDict) -> str:
    weights_bytes = pickle.dumps(weights)
    return base64.b64encode(weights_bytes).decode('utf-8')


def base64_to_weights(base64_str: str) -> OrderedDict:
    weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
    return pickle.loads(weights_bytes)


@app.route('/api/status', methods=['GET'])
def status():
    if fl_server is None:
        return jsonify({"status": "initializing"}), 200

    return jsonify(fl_server.get_server_status())


@app.route('/api/start_round', methods=['POST'])
def start_round():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    result = fl_server.start_round()
    return jsonify(result)


@app.route('/api/get_global_model', methods=['GET'])
def get_global_model():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    weights = fl_server.get_global_model()
    weights_b64 = weights_to_base64(weights)

    xfl_info = fl_server.aggregation_strategy.get_xfl_info()

    dataset_name = fl_config.get('dataset', 'MNIST')
    data_distribution = fl_config.get('dataDistribution', 'iid')

    return jsonify({
        "weights": weights_b64,
        "round": fl_server.current_round,
        "xfl_strategy": xfl_info['strategy'],
        "xfl_param": xfl_info['param'],
        "sparsification_threshold": xfl_info.get('sparsification_threshold', 0.01),
        "quantization_bits": xfl_info.get('quantization_bits', 8),
        "dataset_name": dataset_name,
        "data_distribution": data_distribution
    })


@app.route('/api/submit_update', methods=['POST'])
def submit_update():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.get_json()

    client_id = data.get('client_id')
    weights_b64 = data.get('weights')
    num_samples = data.get('num_samples')
    client_metrics = data.get('metrics', {})
    quantization_meta = data.get('quantization_meta')

    if client_id is None or weights_b64 is None or num_samples is None:
        return jsonify({"error": "Missing required fields"}), 400

    weights = base64_to_weights(weights_b64)

    result = fl_server.submit_client_update(
        client_id=client_id,
        model_weights=weights,
        num_samples=num_samples,
        client_metrics=client_metrics,
        quantization_meta=quantization_meta
    )
    
    return jsonify(result)


@app.route('/api/xfl/set_strategy', methods=['POST'])
def set_xfl_strategy():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.get_json()
    strategy = data.get('strategy', 'all_layers')
    param = data.get('param', 3)
    
    result = fl_server.set_xfl_strategy(strategy, param)
    return jsonify(result)


@app.route('/api/xfl/get_strategy', methods=['GET'])
def get_xfl_strategy():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    xfl_info = fl_server.aggregation_strategy.get_xfl_info()
    return jsonify(xfl_info)


@app.route('/api/metrics/summary', methods=['GET'])
def get_metrics_summary():
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    summary = fl_server.metrics_collector.get_summary_statistics()
    return jsonify(summary)


@app.route('/api/accuracy', methods=['GET'])
def get_accuracy_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number, global_test_accuracy
            FROM round_metrics
            WHERE global_test_accuracy IS NOT NULL
            ORDER BY round_number
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = [row[0] for row in data]
        accuracy = [round(row[1], 2) if row[1] is not None else None for row in data]

        return jsonify({"rounds": rounds, "accuracy": accuracy})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/loss', methods=['GET'])
def get_loss_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number, global_test_loss
            FROM round_metrics
            WHERE global_test_loss IS NOT NULL
            ORDER BY round_number
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = [row[0] for row in data]
        loss = [round(row[1], 4) if row[1] is not None else None for row in data]

        return jsonify({"rounds": rounds, "loss": loss})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clients', methods=['GET'])
def get_clients_data():
    expected_clients = int(fl_config.get('numClients', os.getenv('NUM_CLIENTS', '40')))

    client_data_map = {}
    last_completed_round = 0
    active_in_last_round = set()

    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT client_id,
                   AVG(training_accuracy) as avg_accuracy,
                   AVG(training_time_sec) as avg_time,
                   AVG(cpu_percent) as avg_cpu,
                   AVG(memory_mb) as avg_memory,
                   MAX(round_number) as last_round,
                   COUNT(*) as num_rounds
            FROM client_metrics
            GROUP BY client_id
            ORDER BY client_id
        """)

        data = cursor.fetchall()
        client_data_map = {row[0]: row for row in data}

        cursor.execute("""
                       SELECT MAX(round_number) FROM round_metrics
                       WHERE global_test_accuracy IS NOT NULL
                       """)
        last_completed_round = cursor.fetchone()[0] or 0

        cursor.execute("""
                       SELECT metrics_json FROM round_metrics
                       WHERE round_number = %s AND global_test_accuracy IS NOT NULL
                       """, (last_completed_round,))
        row = cursor.fetchone()
        if row and row[0] and 'submitted_clients' in row[0]:
            active_in_last_round = set(row[0]['submitted_clients'])
        else:
            cursor.execute("""
                           SELECT DISTINCT client_id FROM client_metrics
                           WHERE round_number = %s
                           """, (last_completed_round,))
            active_in_last_round = set(r[0] for r in cursor.fetchall())

        conn.close()
    except Exception as e:
        print(f"WARNING: Database error in /api/clients: {e}")

    current_round = last_completed_round
    round_in_progress = False
    submitted_clients = set()
    selected_clients = set()

    if fl_server is not None:
        server_status = fl_server.get_server_status()
        round_in_progress = server_status.get('round_in_progress', False)
        
        if round_in_progress:
            current_round = server_status.get('current_round', last_completed_round)
            submitted_clients = set(server_status.get('submitted_clients', []))
            selected_clients = set(server_status.get('selected_clients', []))
        else:
            selected_clients = set()

    clients = []
    for client_id in range(expected_clients):
        if round_in_progress:
            if client_id in submitted_clients:
                state = "active"
            elif client_id in selected_clients:
                state = "training"
            else:
                state = "idle"
        else:
            if client_id in active_in_last_round:
                state = "active"
            else:
                state = "idle"

        if client_id in client_data_map:
            row = client_data_map[client_id]
            clients.append({
                "client_id": client_id,
                "avg_accuracy": round(row[1], 2) if row[1] else 0,
                "avg_time": round(row[2], 2) if row[2] else 0,
                "avg_cpu": round(row[3], 2) if row[3] else 0,
                "avg_memory": round(row[4], 2) if row[4] else 0,
                "num_rounds": row[6],
                "state": state
            })
        else:
            clients.append({
                "client_id": client_id,
                "avg_accuracy": 0,
                "avg_time": 0,
                "avg_cpu": 0,
                "avg_memory": 0,
                "num_rounds": 0,
                "state": state
            })

    return jsonify({"clients": clients})


@app.route('/api/bandwidth', methods=['GET'])
def get_bandwidth_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT client_id, AVG(bytes_sent) / 1048576.0 as avg_mb
            FROM client_metrics
            GROUP BY client_id
            ORDER BY client_id
            LIMIT 10
        """)

        data = cursor.fetchall()
        conn.close()

        labels = [f"Pi {str(row[0]).zfill(2)}" for row in data]
        values = [round(row[1], 2) if row[1] else 0 for row in data]

        return jsonify({"labels": labels, "values": values})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/latency', methods=['GET'])
def get_latency_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number, AVG(latency_ms) as avg_latency
            FROM client_metrics
            GROUP BY round_number
            ORDER BY round_number
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = [row[0] for row in data]
        latency = [round(row[1], 2) if row[1] else None for row in data]

        return jsonify({"rounds": rounds, "latency": latency})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/energy', methods=['GET'])
def get_energy_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number, AVG(energy_wh) as avg_energy
            FROM client_metrics
            GROUP BY round_number
            ORDER BY round_number
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = [row[0] for row in data]
        energy = [round(row[1], 4) if row[1] else None for row in data]

        return jsonify({"rounds": rounds, "energy": energy})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/network_metrics', methods=['GET'])
def get_network_metrics_data():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number,
                   AVG(packet_loss_rate) as avg_packet_loss,
                   AVG(jitter_ms) as avg_jitter
            FROM client_metrics
            GROUP BY round_number
            ORDER BY round_number
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = [row[0] for row in data]
        packet_loss = [round(row[1] * 100, 4) if row[1] else None for row in data]
        jitter = [round(row[2], 2) if row[2] else None for row in data]

        return jsonify({"rounds": rounds, "packet_loss": packet_loss, "jitter": jitter})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rounds_history', methods=['GET'])
def get_rounds_history():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT round_number,
                   global_test_accuracy,
                   global_test_loss,
                   aggregation_time_sec,
                   num_clients,
                   strategy,
                   metrics_json
            FROM round_metrics
            ORDER BY round_number DESC
            LIMIT 50
        """)

        data = cursor.fetchall()
        conn.close()

        rounds = []
        for row in data:
            is_completed = row[1] is not None
            
            rounds.append({
                "round": row[0],
                "accuracy": round(row[1], 2) if row[1] else None,
                "loss": round(row[2], 4) if row[2] else None,
                "agg_time": round(row[3], 2) if row[3] else None,
                "clients": row[4],
                "strategy": row[5] if row[5] else 'all_layers',
                "status": "completed" if is_completed else "in_progress"
            })

        return jsonify({"rounds": rounds})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history_by_strategy', methods=['GET'])
def get_history_by_strategy():
    """
    Get history data grouped by strategy and sessions.

    ── CHANGED: sessions are now real docker-compose up/down boundaries ──────
    Each distinct (strategy, session_id) pair becomes one "experiment" in the
    History page.  No more round-number gap detection.

    Returns:
    - List of strategies used
    - For each strategy: list of sessions (experiments)
    - Each session: rounds data, accuracy/loss evolution, config used,
      started_at timestamp from server_sessions
    """
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        # Get all distinct strategies that have at least one completed round
        cursor.execute("""
            SELECT DISTINCT strategy 
            FROM round_metrics 
            WHERE strategy IS NOT NULL AND global_test_accuracy IS NOT NULL
            ORDER BY strategy
        """)
        strategies = [row[0] for row in cursor.fetchall()]
        
        if not strategies:
            conn.close()
            return jsonify({"strategies": [], "message": "No completed rounds found"})

        # Build a lookup of session metadata (started_at, hostname)
        cursor.execute("SELECT session_id, started_at, hostname FROM server_sessions")
        session_meta = {row[0]: {"started_at": row[1], "hostname": row[2]} for row in cursor.fetchall()}

        result = []
        
        for strategy in strategies:
            # Get all distinct session_ids for this strategy, ordered by first round timestamp
            cursor.execute("""
                           SELECT rm.session_id
                           FROM round_metrics rm
                           LEFT JOIN server_sessions ss ON ss.session_id = rm.session_id
                           WHERE rm.strategy = %s
                           AND rm.global_test_accuracy IS NOT NULL
                           AND rm.session_id IS NOT NULL
                           GROUP BY rm.session_id, ss.started_at
                           ORDER BY ss.started_at ASC NULLS LAST
                           """, (strategy,))
            session_ids = [row[0] for row in cursor.fetchall()]

            # Also handle legacy rows that have no session_id (NULL) – treat them
            # as a single legacy "session" so old data is not lost
            cursor.execute("""
                SELECT COUNT(*)
                FROM round_metrics
                WHERE strategy = %s AND global_test_accuracy IS NOT NULL AND session_id IS NULL
            """, (strategy,))
            legacy_count = cursor.fetchone()[0]

            experiments = []
            experiment_counter = 1

            # ── Process NULL-session (legacy) rows first ───────────────────
            if legacy_count > 0:
                cursor.execute("""
                    SELECT round_number, global_test_accuracy, global_test_loss,
                           aggregation_time_sec, num_clients, timestamp, metrics_json
                    FROM round_metrics
                    WHERE strategy = %s AND global_test_accuracy IS NOT NULL AND session_id IS NULL
                    ORDER BY round_number
                """, (strategy,))
                rows = cursor.fetchall()

                exp = _build_experiment(rows, experiment_counter, session_id=None,
                                        session_meta=session_meta, cursor=cursor)
                exp["is_legacy"] = True
                experiments.append(exp)
                experiment_counter += 1

            # ── Process each real session ──────────────────────────────────
            for sid in session_ids:
                cursor.execute("""
                    SELECT round_number, global_test_accuracy, global_test_loss,
                           aggregation_time_sec, num_clients, timestamp, metrics_json
                    FROM round_metrics
                    WHERE strategy = %s AND global_test_accuracy IS NOT NULL AND session_id = %s
                    ORDER BY round_number
                """, (strategy, sid))
                rows = cursor.fetchall()

                if not rows:
                    continue

                exp = _build_experiment(rows, experiment_counter, session_id=sid,
                                        session_meta=session_meta, cursor=cursor)
                exp["is_legacy"] = False
                experiments.append(exp)
                experiment_counter += 1

            result.append({
                "strategy": strategy,
                "experiments": experiments,
                "total_experiments": len(experiments),
                "total_rounds": sum(len(e["rounds"]) for e in experiments)
            })

        conn.close()
        
        return jsonify({"strategies": result})

    except Exception as e:
        print(f"Error in /api/history_by_strategy: {e}")
        return jsonify({"error": str(e)}), 500


def _build_experiment(rows, experiment_id: int, session_id, session_meta: dict, cursor) -> dict:
    """
    Helper: build an experiment dict from a list of DB rows for one session.
    Each row: (round_number, accuracy, loss, agg_time, num_clients, timestamp, metrics_json)
    """
    accuracy_evolution = []
    loss_evolution = []
    rounds_list = []

    for row in rows:
        round_num, accuracy, loss, agg_time, num_clients, timestamp, metrics_json = row

        accuracy = round(accuracy, 2) if accuracy is not None else None
        loss = round(loss, 4) if loss is not None else None
        agg_time = round(agg_time, 2) if agg_time is not None else None

        rounds_list.append({
            "round": round_num,
            "accuracy": accuracy,
            "loss": loss,
            "agg_time": agg_time,
            "clients": num_clients,
            "timestamp": timestamp
        })

        if accuracy is not None:
            accuracy_evolution.append({"round": round_num, "accuracy": accuracy})
        if loss is not None:
            loss_evolution.append({"round": round_num, "loss": loss})

    # Statistics
    accuracies = [r["accuracy"] for r in rounds_list if r["accuracy"] is not None]
    losses = [r["loss"] for r in rounds_list if r["loss"] is not None]
    agg_times = [r["agg_time"] for r in rounds_list if r["agg_time"] is not None]

    stats = {
        "total_rounds": len(rounds_list),
        "best_accuracy": max(accuracies) if accuracies else None,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "avg_agg_time": sum(agg_times) / len(agg_times) if agg_times else None,
        "first_round": rounds_list[0]["round"] if rounds_list else 0,
        "last_round": rounds_list[-1]["round"] if rounds_list else 0
    }

    # Session metadata
    meta = session_meta.get(session_id, {}) if session_id else {}
    started_at = meta.get("started_at")
    hostname = meta.get("hostname")

    # Config snapshot (latest values – same as before)
    cursor.execute("SELECT config_key, config_value FROM fl_config ORDER BY updated_at")
    config_rows = cursor.fetchall()
    config_dict = {}
    for key, value in config_rows:
        if value and value.isdigit():
            config_dict[key] = int(value)
        else:
            try:
                config_dict[key] = float(value)
            except:
                config_dict[key] = value

    return {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "started_at": started_at,
        "hostname": hostname,
        "rounds": rounds_list,
        "accuracy_evolution": accuracy_evolution,
        "loss_evolution": loss_evolution,
        "stats": stats,
        "config": config_dict
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM round_metrics) as total_rounds,
                (SELECT COUNT(*) FROM client_metrics) as total_client_submissions,
                (SELECT COUNT(*) FROM round_metrics WHERE global_test_accuracy IS NOT NULL) as completed_rounds
        """)
        stats = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "session_id": CURRENT_SESSION_ID,
            "stats": {
                "total_rounds": stats[0],
                "total_client_submissions": stats[1],
                "completed_rounds": stats[2]
            }
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route('/api/dse/sweep', methods=['POST'])
def dse_sweep():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400

        session_id = start_dse_sweep(data)
        print(f"DSE sweep started: {session_id}")
        return jsonify({"session_id": session_id, "status": "started"})
    except Exception as e:
        print(f"DSE sweep error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/dse/sessions', methods=['GET'])
def dse_sessions():
    try:
        sessions = list_dse_sessions()
        return jsonify({"sessions": sessions})
    except Exception as e:
        print(f"DSE sessions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/dse/results/<session_id>', methods=['GET'])
def dse_results(session_id):
    try:
        result = load_dse_session(session_id)
        if result is None:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(result)
    except Exception as e:
        print(f"DSE results error for {session_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/dse/status/<session_id>', methods=['GET'])
def dse_status(session_id):
    try:
        status = get_dse_job_status(session_id)
        return jsonify({"session_id": session_id, "status": status})
    except Exception as e:
        print(f"DSE status error for {session_id}: {e}")
        return jsonify({"error": str(e)}), 500


def create_server(
    global_model: torch.nn.Module,
    test_loader=None,
    aggregation_strategy: str = "fedavg",
    num_rounds: int = 10,
    clients_per_round: int = 5,
    db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics",
    xfl_strategy: str = "all_layers",
    xfl_param: int = 3
) -> FLServer:
    """Create and initialize FL server with XFL support"""
    global fl_server

    fl_server = FLServer(
        global_model=global_model,
        aggregation_strategy=aggregation_strategy,
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        db_url=db_url,
        xfl_strategy=xfl_strategy,
        xfl_param=xfl_param
    )
    
    fl_server.test_loader = test_loader
    
    try:
        saved_config = load_fl_config_from_db()
        if saved_config:
            for key, value in saved_config.items():
                if key not in ['numClients']:
                    fl_config[key] = value
            print("Loaded config from database")
            
            if 'clientsPerRound' in saved_config:
                fl_server.clients_per_round = int(saved_config['clientsPerRound'])
                print(f"Updated clients_per_round to {fl_server.clients_per_round} from saved config")
            
            if 'numRounds' in saved_config:
                old_num_rounds = fl_server.num_rounds
                fl_server.num_rounds = int(saved_config['numRounds'])
                print(f"Updated numRounds from {old_num_rounds} to {fl_server.num_rounds} from saved config")
            
            if 'strategy' in saved_config:
                strategy = saved_config.get('strategy', 'all_layers')
                param = saved_config.get('xflParam', 3)
                print(f"Loading strategy from config: {strategy}, param: {param}")
                fl_server.set_xfl_strategy(strategy, param)
    except Exception as e:
        print(f"WARNING:  Warning: Could not load config: {e}")
    
    return fl_server


def run_server(host: str = "localhost", port: int = 5000, debug: bool = False):
    """Run Flask server"""
    print(f"\nSTARTING: Starting FL Server on {host}:{port}")
    print(f"   Session ID: {CURRENT_SESSION_ID}")
    app.run(host=host, port=port, debug=debug, threaded=True)