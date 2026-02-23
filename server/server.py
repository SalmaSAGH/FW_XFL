"""
Flask server for Federated Learning with XFL support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import base64
import hashlib
import secrets
import os
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
import gc

# Database URL for dashboard APIs
# Default to localhost for local development, Docker will override via environment variable
DB_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:newpassword@localhost:5432/xfl_metrics')

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
            # Parse the database URL to get connection parameters
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
                maxconn=10,
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname
            )
            print("✅ Server connection pool initialized (min: 2, max: 10)")
            return _connection_pool
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize connection pool: {e}")
            return None


def _get_connection():
    """Get a connection from the pool"""
    global _connection_pool
    
    if _connection_pool is None:
        _init_connection_pool()
    
    if _connection_pool:
        try:
            conn = _connection_pool.getconn()
            # Verify the connection is still valid
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except:
                # Connection is invalid, try to get a new one
                _connection_pool.putconn(conn, close=True)
                conn = _connection_pool.getconn()
                return conn
        except Exception as e:
            print(f"⚠️ Warning: Could not get connection from pool: {e}")
    
    # Fallback: create a new connection
    return psycopg2.connect(DB_URL)


def _return_connection(conn):
    """Return a connection to the pool"""
    global _connection_pool
    
    if _connection_pool and conn:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            print(f"⚠️ Warning: Could not return connection to pool: {e}")
            try:
                conn.close()
            except:
                pass
    elif conn:
        try:
            conn.close()
        except:
            pass


def init_fl_config_database():
    """Initialize FL config table in the database"""
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    # FL Config table - stores FL configuration and server state
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
    print("✅ FL config table initialized")


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
        
        # Save each config key as a separate row
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
            # Try to convert to appropriate type
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

    # FL Server State table - stores server state like current_round
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
    print("✅ FL server state table initialized")


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

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at REAL NOT NULL
        )
    """)

    # Sessions table
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
    print("✅ Users and sessions tables initialized")


def get_user_from_db(username: str):
    """Get user from database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT username, password_hash, created_at FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                'username': result[0],
                'password': result[1],
                'created_at': result[2]
            }
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None


def create_user_in_db(username: str, password_hash: str):
    """Create user in database"""
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
    """Create session in database"""
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
    """Get session from database"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT token, username, created_at FROM sessions WHERE token = %s", (token,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                'token': result[0],
                'username': result[1],
                'created_at': result[2]
            }
        return None
    except Exception as e:
        print(f"Error getting session: {e}")
        return None


def delete_session_from_db(token: str):
    """Delete session from database"""
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
        """
        Args:
            global_model: Initial global model
            aggregation_strategy: Aggregation method ('fedavg', 'fedprox')
            num_rounds: Total number of FL rounds
            clients_per_round: Number of clients per round
            db_url: PostgreSQL database URL
            xfl_strategy: XFL layer selection strategy
            xfl_param: XFL parameter (e.g., N for first_n/last_n)
        """
        self.global_model = global_model
        self.aggregation_strategy = create_aggregation_strategy(
            aggregation_strategy,
            xfl_strategy=xfl_strategy,
            xfl_param=xfl_param
        )
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        
        # Metrics collector
        self.metrics_collector = ServerMetricsCollector(db_url)
        
        # State management
        # Load current_round from database if available
        self.current_round = self._load_current_round()
        self.round_in_progress = False
        self.client_submissions = []  # List of submission dicts
        self.selected_clients = []  # Clients selected for current round
        self.lock = threading.Lock()
        
        # Test metrics
        self.test_loader = None
        
        print(f"✅ FLServer initialized")
        print(f"   Strategy: {self.aggregation_strategy.name}")
        print(f"   XFL: {xfl_strategy}")
        print(f"   Total rounds: {self.num_rounds}")
        print(f"   Clients per round: {self.clients_per_round}")
        print(f"   Current round (loaded from DB): {self.current_round}")
    
    def _load_current_round(self) -> int:
        """Load current round from database"""
        try:
            # Try to get from server state first
            saved_round = get_server_state('current_round')
            
            if saved_round:
                # Check if the saved round has corresponding metrics in round_metrics
                # If there's no metrics for this round, it might be an incomplete round
                conn = psycopg2.connect(DB_URL)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT global_test_accuracy, metrics_json 
                    FROM round_metrics 
                    WHERE round_number = %s
                """, (int(saved_round),))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    # Check if this is an incomplete round (no accuracy yet)
                    accuracy = result[0]
                    metrics_json = result[1]
                    
                    if accuracy is None or (metrics_json and metrics_json.get('status') == 'in_progress'):
                        # This is an incomplete round - fall back to previous completed round
                        conn = psycopg2.connect(DB_URL)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT MAX(round_number) 
                            FROM round_metrics 
                            WHERE global_test_accuracy IS NOT NULL
                        """)
                        max_completed = cursor.fetchone()
                        conn.close()
                        
                        if max_completed and max_completed[0]:
                            print(f"⚠️  Warning: Round {saved_round} appears incomplete, using round {max_completed[0]}")
                            return max_completed[0]
                
                return int(saved_round)
            
            # Fallback: get max round from round_metrics
            conn = psycopg2.connect(DB_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(round_number) FROM round_metrics")
            result = cursor.fetchone()
            conn.close()
            if result and result[0]:
                return result[0]
        except Exception as e:
            print(f"⚠️  Warning: Could not load current round from database: {e}")
        return 0
    
    def _save_current_round(self):
        """Save current round to database"""
        try:
            save_server_state('current_round', str(self.current_round))
        except Exception as e:
            print(f"⚠️  Warning: Could not save current round to database: {e}")
    
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

            # Reload config from database to get latest values (fix for config not being synced)
            try:
                latest_config = load_fl_config_from_db()
                if latest_config:
                    # Update fl_config with latest values from database
                    for key, value in latest_config.items():
                        fl_config[key] = value
                    
                    # Update server's clients_per_round from config if explicitly set
                    if 'clientsPerRound' in latest_config:
                        self.clients_per_round = int(latest_config['clientsPerRound'])
                        print(f"✅ Using clientsPerRound from config: {self.clients_per_round}")
                    
                    # Update num_rounds if set in config
                    if 'numRounds' in latest_config:
                        self.num_rounds = int(latest_config['numRounds'])
                        print(f"✅ Using numRounds from config: {self.num_rounds}")
            except Exception as e:
                print(f"⚠️  Warning: Could not reload config from database: {e}")

            self.current_round += 1
            self.round_in_progress = True
            self.client_submissions = []  # Reset submissions list

            # Save current round to database for persistence
            self._save_current_round()

            # Save placeholder round metrics when round starts (to preserve data on restart)
            # This ensures the round is tracked even if server restarts before aggregation
            try:
                self.metrics_collector.store_round_metrics(
                    round_number=self.current_round,
                    num_clients=0,  # Will be updated after aggregation
                    aggregation_time=0,  # Will be updated after aggregation
                    global_test_loss=None,  # Will be updated after aggregation
                    global_test_accuracy=None,  # Will be updated after aggregation
                    total_samples=0,  # Will be updated after aggregation
                    additional_metrics={"status": "in_progress", "clients_expected": self.clients_per_round}
                )
                print(f"✅ Round {self.current_round} metrics placeholder saved")
            except Exception as e:
                print(f"⚠️  Warning: Could not save round metrics placeholder: {e}")

            # Select clients for this round (for dashboard display)
            # Use the configured numClients from fl_config
            import random
            total_clients = int(fl_config.get('numClients', 40))
            
            # Ensure clients_per_round doesn't exceed total_clients
            actual_clients_per_round = min(self.clients_per_round, total_clients)
            
            self.selected_clients = random.sample(range(total_clients), actual_clients_per_round)

            # Get XFL info for logging
            xfl_info = self.aggregation_strategy.get_xfl_info()
            print(f"\n🔄 Starting Round {self.current_round}/{self.num_rounds}")
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

            # Check for duplicate submission from same client
            if any(sub["client_id"] == client_id for sub in self.client_submissions):
                return {
                    "status": "duplicate",
                    "message": f"Client {client_id} has already submitted for this round"
                }

            # Store submission
            self.client_submissions.append({
                "client_id": client_id,
                "weights": model_weights,
                "num_samples": num_samples,
                "metrics": client_metrics,
                "quantization_meta": quantization_meta
            })

            # Store client metrics in database
            self.metrics_collector.store_client_metrics(
                self.current_round,
                client_id,
                client_metrics
            )

            print(f"   ✅ Received update from Client {client_id} "
                  f"({len(self.client_submissions)}/{self.clients_per_round})")

            # Check if all clients submitted
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
        print(f"\n📊 Aggregating {len(self.client_submissions)} clients with XFL: {xfl_info['strategy']}")

        start_time = time.time()

        try:
            # Extract weights and sample counts
            client_weights = [sub["weights"] for sub in self.client_submissions]
            client_num_samples = [sub["num_samples"] for sub in self.client_submissions]

            # Check if any quantization metadata is present (for XFL quantization)
            has_quantization = any(sub.get("quantization_meta") for sub in self.client_submissions)

            if has_quantization and xfl_info['variant'] == 'quantization':
                print("   🔧 Applying dequantization before aggregation...")
                # Dequantize weights before aggregation
                client_weights = self._dequantize_client_weights(client_weights)

            # Aggregate with XFL
            aggregated_weights = self.aggregation_strategy.aggregate(
                client_weights,
                client_num_samples
            )

            # Update global model
            if isinstance(self.aggregation_strategy, XFL):
                # For XFL, aggregated_weights contains only updated layers
                current_state = self.global_model.state_dict()
                current_state.update(aggregated_weights)
                self.global_model.load_state_dict(current_state)
            else:
                # For FedAvg variants, aggregated_weights contains all layers
                self.global_model.load_state_dict(aggregated_weights)

            aggregation_time = time.time() - start_time

            # Evaluate global model
            test_loss, test_accuracy = None, None
            if self.test_loader is not None:
                try:
                    test_loss, test_accuracy = self._evaluate_global_model()
                except Exception as e:
                    print(f"⚠️  Global model evaluation failed: {e}")
                    test_loss, test_accuracy = None, None

            # Store round metrics with the strategy used
            xfl_info = self.aggregation_strategy.get_xfl_info()
            self.metrics_collector.store_round_metrics(
                round_number=self.current_round,
                num_clients=len(self.client_submissions),
                aggregation_time=aggregation_time,
                global_test_loss=test_loss,
                global_test_accuracy=test_accuracy,
                total_samples=sum(client_num_samples),
                strategy=xfl_info['strategy']
            )

            print(f"✅ Round {self.current_round} completed in {aggregation_time:.2f}s")
            if test_accuracy is not None:
                print(f"   Global Test Accuracy: {test_accuracy:.2f}%")

        except Exception as e:
            aggregation_time = time.time() - start_time
            print(f"❌ Aggregation failed for round {self.current_round}: {e}")
            print(f"   Round will be completed anyway to prevent blocking")

        # Save current round to database for persistence (backup after aggregation)
        self._save_current_round()

        # Reset round state
        self.round_in_progress = False
        self.client_submissions = []  # Clear submissions after aggregation
        
        # Force garbage collection to free memory after aggregation
        # This is especially important when handling large model weights from many clients
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
        """
        Dequantize client weights before aggregation for XFL quantization
        """
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
        """Get current global model weights"""
        with self.lock:
            return self.global_model.state_dict()
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status with XFL info"""
        with self.lock:
            xfl_info = self.aggregation_strategy.get_xfl_info()
            layer_names = list(self.global_model.state_dict().keys())
            num_layers = len(set(name.split('.')[0] for name in layer_names))
            
            # Get total_clients from fl_config - this is the TOTAL number of clients available in the system
            # NOT the number of clients that participated in the last round
            total_clients = int(fl_config.get('numClients', 40))
            latest_accuracy = None
            
            try:
                conn = psycopg2.connect(DB_URL)
                cursor = conn.cursor()
                
                # Get latest accuracy from completed rounds
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
                print(f"⚠️ Warning: Could not get additional status from database: {e}")
            
            return {
                "current_round": self.current_round,
                "total_rounds": self.num_rounds,
                "round_in_progress": self.round_in_progress,
                "submissions_received": len(self.client_submissions),
                "clients_expected": self.clients_per_round,
                "total_clients": total_clients,
                "latest_accuracy": latest_accuracy,
                "submitted_clients": [sub["client_id"] for sub in self.client_submissions],
                "selected_clients": self.selected_clients if self.round_in_progress else [],
                "xfl_strategy": xfl_info['strategy'],
                "xfl_param": xfl_info['param'],
                "num_layers": num_layers
            }
    
    def set_xfl_strategy(self, strategy: str, param: int = 3) -> Dict[str, Any]:
        """Change XFL strategy dynamically"""
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

# Secret key for sessions
app.secret_key = 'xfl-rpilab-secret-key-change-in-production'

# Global server instance
fl_server: FLServer = None

# Global configuration storage (default values)
fl_config = {
    "numClients": 40,
    "numRounds": 50,
    "dataset": "MNIST",
    "dataDistribution": "iid",
    "strategy": "all_layers",
    "xflParam": 3,
    "localEpochs": 2,
    "batchSize": 512,
    "learningRate": 0.01,
}

# Initialize database at startup
try:
    init_user_database()
    init_fl_config_database()
    init_server_state_database()
except Exception as e:
    print(f"⚠️  Database initialization warning: {e}")
    print("   Server will continue but auth features may not work properly")

# Load FL config from database at startup
try:
    loaded_config = load_fl_config_from_db()
    if loaded_config:
        fl_config.update(loaded_config)
        print("✅ FL config loaded from database")
        
        # Note: We no longer update clients_per_round from numClients at startup
        # The server should use the clients_per_round value from its initialization
        # or from the clientsPerRound config key if explicitly set
except Exception as e:
    print(f"⚠️  Warning: Could not load FL config from database: {e}")

# Simple user storage (in production, use a proper database)
users_db = {}

# In-memory session storage
sessions = {}


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save FL configuration"""
    global fl_config
    data = request.get_json()
    
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    # Update configuration
    fl_config.update(data)
    
    # Save to database for persistence
    save_fl_config_to_db(fl_config)
    
    # Apply strategy if changed
    if fl_server is not None and 'strategy' in data:
        strategy = data.get('strategy', 'all_layers')
        param = data.get('xflParam', 3)
        fl_server.set_xfl_strategy(strategy, param)
    
    # Note: We no longer update clients_per_round from numClients
    # numClients = total number of clients available (from docker-compose)
    # clients_per_round = number of clients to select per round (default 5)
    # These are two separate settings!
    
    # If user explicitly sets clientsPerRound, update it
    if fl_server is not None and 'clientsPerRound' in data:
        fl_server.clients_per_round = int(data['clientsPerRound'])
        print(f"✅ Updated clients_per_round to {fl_server.clients_per_round}")
    
    return jsonify({"status": "success", "message": "Configuration saved", "config": fl_config})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current FLLet me apply the edit configuration"""
    return jsonify({"config": fl_config})


@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
        
        # Check if user already exists in database
        existing_user = get_user_from_db(username)
        if existing_user:
            return jsonify({"error": "Username already exists"}), 409
        
        # Simple password hashing
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Create user in database
        success = create_user_in_db(username, hashed_password)
        if not success:
            return jsonify({"error": "Registration failed. Please try again."}), 500
        
        return jsonify({"status": "success", "message": "User registered successfully"}), 201
    except Exception as e:
        print(f"Error in register: {e}")
        return jsonify({"error": "Registration failed. Please try again."}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    # Get user from database
    user = get_user_from_db(username)
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if user['password'] != hashed_password:
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Create session in database
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
    """Logout user"""
    data = request.get_json()
    token = data.get('token')
    
    if token:
        delete_session_from_db(token)
    
    return jsonify({"status": "success", "message": "Logged out successfully"}), 200


@app.route('/api/verify_token', methods=['POST'])
def verify_token():
    """Verify if a token is valid"""
    data = request.get_json()
    token = data.get('token')
    
    session = get_session_from_db(token)
    if session:
        return jsonify({
            "status": "valid",
            "username": session['username']
        }), 200
    
    return jsonify({"status": "invalid"}), 401


def weights_to_base64(weights: OrderedDict) -> str:
    """Serialize weights to base64 string"""
    weights_bytes = pickle.dumps(weights)
    return base64.b64encode(weights_bytes).decode('utf-8')


def base64_to_weights(base64_str: str) -> OrderedDict:
    """Deserialize weights from base64 string"""
    weights_bytes = base64.b64decode(base64_str.encode('utf-8'))
    return pickle.loads(weights_bytes)


@app.route('/api/status', methods=['GET'])
def status():
    """Get server status with XFL info"""
    if fl_server is None:
        return jsonify({"status": "initializing"}), 200

    return jsonify(fl_server.get_server_status())


@app.route('/api/start_round', methods=['POST'])
def start_round():
    """Start a new FL round"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    result = fl_server.start_round()
    return jsonify(result)


@app.route('/api/get_global_model', methods=['GET'])
def get_global_model():
    """Get current global model weights"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    weights = fl_server.get_global_model()
    weights_b64 = weights_to_base64(weights)

    xfl_info = fl_server.aggregation_strategy.get_xfl_info()

    return jsonify({
        "weights": weights_b64,
        "round": fl_server.current_round,
        "xfl_strategy": xfl_info['strategy'],
        "xfl_param": xfl_info['param'],
        "sparsification_threshold": xfl_info.get('sparsification_threshold', 0.01),
        "quantization_bits": xfl_info.get('quantization_bits', 8)
    })


@app.route('/api/submit_update', methods=['POST'])
def submit_update():
    """Receive client model update"""
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
    """Set XFL strategy dynamically"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.get_json()
    strategy = data.get('strategy', 'all_layers')
    param = data.get('param', 3)
    
    result = fl_server.set_xfl_strategy(strategy, param)
    return jsonify(result)


@app.route('/api/xfl/get_strategy', methods=['GET'])
def get_xfl_strategy():
    """Get current XFL strategy"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    xfl_info = fl_server.aggregation_strategy.get_xfl_info()
    return jsonify(xfl_info)


@app.route('/api/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get metrics summary"""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    summary = fl_server.metrics_collector.get_summary_statistics()
    return jsonify(summary)


@app.route('/api/accuracy', methods=['GET'])
def get_accuracy_data():
    """Get accuracy data for plotting"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        # Only get completed rounds (where global_test_accuracy is NOT NULL)
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

        return jsonify({
            "rounds": rounds,
            "accuracy": accuracy
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/loss', methods=['GET'])
def get_loss_data():
    """Get loss data for plotting"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        # Only get completed rounds (where global_test_loss is NOT NULL)
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

        return jsonify({
            "rounds": rounds,
            "loss": loss
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clients', methods=['GET'])
def get_clients_data():
    """Get client metrics"""
    # Use numClients from fl_config, fallback to environment variable or default 40
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

        cursor.execute("SELECT MAX(round_number) FROM round_metrics")
        last_completed_round = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT DISTINCT client_id
            FROM client_metrics
            WHERE round_number = %s
        """, (last_completed_round,))
        active_in_last_round = set(row[0] for row in cursor.fetchall())

        conn.close()
    except Exception as e:
        print(f"⚠️ Database error in /api/clients: {e}")

    current_round = last_completed_round
    round_in_progress = False
    submitted_clients = set()
    selected_clients = set()

    if fl_server is not None:
        server_status = fl_server.get_server_status()
        round_in_progress = server_status.get('round_in_progress', False)
        
        # Debug logging
        print(f"[DEBUG] /api/clients - round_in_progress: {round_in_progress}")
        
        if round_in_progress:
            current_round = server_status.get('current_round', last_completed_round)
            submitted_clients = set(server_status.get('submitted_clients', []))
            selected_clients = set(server_status.get('selected_clients', []))
            
            print(f"[DEBUG] /api/clients - current_round: {current_round}")
            print(f"[DEBUG] /api/clients - submitted_clients: {submitted_clients}")
            print(f"[DEBUG] /api/clients - selected_clients: {selected_clients}")
        else:
            selected_clients = set()

    clients = []
    for client_id in range(expected_clients):
        if round_in_progress:
            if client_id in submitted_clients:
                # Client has submitted this round - show as active (green)
                state = "active"
            elif client_id in selected_clients:
                # Client is selected for this round but hasn't submitted yet - show as training (blue)
                state = "training"
            else:
                # Client is not selected for this round - show as idle (gray)
                state = "idle"
        else:
            # No round in progress - show last round's active clients or idle
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
    """Get bandwidth usage per client"""
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

        return jsonify({
            "labels": labels,
            "values": values
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/latency', methods=['GET'])
def get_latency_data():
    """Get latency data for plotting"""
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

        return jsonify({
            "rounds": rounds,
            "latency": latency
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/energy', methods=['GET'])
def get_energy_data():
    """Get energy consumption data for plotting"""
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

        return jsonify({
            "rounds": rounds,
            "energy": energy
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/network_metrics', methods=['GET'])
def get_network_metrics_data():
    """Get network metrics (packet loss and jitter) for plotting"""
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

        return jsonify({
            "rounds": rounds,
            "packet_loss": packet_loss,
            "jitter": jitter
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rounds_history', methods=['GET'])
def get_rounds_history():
    """Get detailed history of all rounds (completed and in-progress)"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        # Get ALL rounds - both completed and in-progress
        # This ensures the frontend always gets data, even if no rounds are completed yet
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
            # Determine if round is completed or in-progress
            is_completed = row[1] is not None  # global_test_accuracy is not null
            
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify database connectivity"""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        # Get database info
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
            "stats": {
                "total_rounds": stats[0],
                "total_client_submissions": stats[1],
                "completed_rounds": stats[2]
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


def create_server(
    global_model: torch.nn.Module,
    test_loader = None,
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
    
    # Load saved config from database and update fl_config
    # We no longer update clients_per_round from numClients because:
    # - numClients = total number of clients available (from docker-compose)
    # - clients_per_round = number of clients to select per round (default 5)
    # These are two separate settings!
    try:
        saved_config = load_fl_config_from_db()
        if saved_config:
            # Update global fl_config with saved values
            for key, value in saved_config.items():
                if key not in ['numClients']:  # Don't update these from DB
                    fl_config[key] = value
            print("✅ Loaded config from database")
            
            # If user has explicitly set clientsPerRound in config, use it
            if 'clientsPerRound' in saved_config:
                fl_server.clients_per_round = int(saved_config['clientsPerRound'])
                print(f"✅ Updated clients_per_round to {fl_server.clients_per_round} from saved config")
            
            # CRITICAL FIX: Load and apply the XFL strategy from saved config on startup
            # This ensures the strategy persists across server restarts
            if 'strategy' in saved_config:
                strategy = saved_config.get('strategy', 'all_layers')
                param = saved_config.get('xflParam', 3)
                print(f"✅ Loading strategy from config: {strategy}, param: {param}")
                fl_server.set_xfl_strategy(strategy, param)
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
    
    return fl_server


def run_server(host: str = "localhost", port: int = 5000, debug: bool = False):
    """Run Flask server"""
    print(f"\n🚀 Starting FL Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
