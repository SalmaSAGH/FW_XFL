"""
Real-time dashboard for monitoring FL experiments with XFL support
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import json
from pathlib import Path
import threading
import time
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_parser import load_config


class DashboardServer:
    """
    Real-time dashboard server for FL monitoring with XFL
    """

    def __init__(self, db_url: str = "postgresql://postgres:newpassword@postgres:5432/xfl_metrics", port: int = 5001):
        """
        Args:
            db_url: PostgreSQL database URL
            port: Port for dashboard server
        """
        self.db_url = db_url
        self.port = port
        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        CORS(self.app)

        # Load configuration
        try:
            self.config = load_config()
            print(f"✅ Configuration loaded: {self.config.federated_learning.num_clients} clients, {self.config.federated_learning.num_rounds} rounds")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            self.config = None

        # Caching for performance
        self._server_status_cache = None
        self._cache_timestamp = 0
        self._cache_timeout = 0  # No caching for real-time updates

        # Setup routes
        self._setup_routes()

        print(f"✅ Dashboard server initialized on port {port}")
        print(f"   XFL strategy control enabled")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current experiment status with XFL info"""
            try:
                # Use NUM_CLIENTS environment variable for expected clients
                expected_clients = int(os.getenv('NUM_CLIENTS', '40'))
                # Use config values for rounds if available, otherwise fall back to environment
                if self.config:
                    total_rounds_config = self.config.federated_learning.num_rounds
                else:
                    total_rounds_config = int(os.getenv('NUM_ROUNDS', '50'))

                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()

                cursor.execute("SELECT MAX(round_number) FROM round_metrics")
                max_completed_round = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT COUNT(DISTINCT client_id)
                    FROM client_metrics
                    WHERE round_number = %s
                """, (max_completed_round,))
                active_clients_last_round = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT round_number, global_test_accuracy, global_test_loss
                    FROM round_metrics
                    ORDER BY round_number DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()

                # Default values
                round_in_progress = False
                current_round = max_completed_round
                submissions_received = 0

                # Detect round in progress from database if server not available
                if latest and latest[1] is None:
                    round_in_progress = True
                    current_round = latest[0]
                    cursor.execute("SELECT DISTINCT client_id FROM client_metrics WHERE round_number = %s", (current_round,))
                    submitted_clients_temp = cursor.fetchall()
                    submissions_received = len(submitted_clients_temp)

                # For xfl, try to fetch from server, else default
                xfl_strategy = "all_layers"
                xfl_param = 3
                num_layers = 0
                server_status = None
                submitted_clients = []
                selected_clients = []
                try:
                    import requests
                    server_status = requests.get('http://server:5000/status', timeout=10).json()
                    # Override with server status if available
                    round_in_progress = server_status.get('round_in_progress', round_in_progress)
                    current_round = server_status.get('current_round', current_round)
                    submissions_received = server_status.get('submissions_received', submissions_received)
                    xfl_strategy = server_status.get('xfl_strategy', 'all_layers')
                    xfl_param = server_status.get('xfl_param', 3)
                    num_layers = server_status.get('num_layers', 0)
                    submitted_clients = server_status.get('submitted_clients', [])
                    selected_clients = server_status.get('selected_clients', [])
                except:
                    pass

                conn.close()

                return jsonify({
                    "current_round": current_round,
                    "total_rounds": total_rounds_config,  # Use configured total rounds
                    "total_clients": active_clients_last_round if active_clients_last_round > 0 else expected_clients,
                    "latest_accuracy": round(latest[1], 2) if latest and latest[1] else None,
                    "latest_loss": round(latest[2], 4) if latest and latest[2] else None,
                    "round_in_progress": round_in_progress,
                    "clients_expected": expected_clients,
                    "submissions_received": submissions_received,
                    "xfl_strategy": xfl_strategy,
                    "xfl_param": xfl_param,
                    "num_layers": num_layers,
                    "submitted_clients": submitted_clients,
                    "selected_clients": selected_clients
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/accuracy')
        def get_accuracy_data():
            """Get accuracy data for plotting"""
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT round_number, global_test_accuracy
                    FROM round_metrics
                    ORDER BY round_number
                """)

                data = cursor.fetchall()
                conn.close()

                rounds = [row[0] for row in data]
                accuracy = [round(row[1], 2) if row[1] else None for row in data]

                return jsonify({
                    "rounds": rounds,
                    "accuracy": accuracy
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/loss')
        def get_loss_data():
            """Get loss data for plotting"""
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT round_number, global_test_loss
                    FROM round_metrics
                    ORDER BY round_number
                """)

                data = cursor.fetchall()
                conn.close()

                rounds = [row[0] for row in data]
                loss = [round(row[1], 4) if row[1] else None for row in data]

                return jsonify({
                    "rounds": rounds,
                    "loss": loss
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/clients')
        def get_clients_data():
            """Get client metrics"""
            # Use NUM_CLIENTS environment variable for expected clients (consistent with /api/status)
            expected_clients = int(os.getenv('NUM_CLIENTS', '40'))

            client_data_map = {}
            last_completed_round = 0
            active_in_last_round = set()

            # Try to fetch data from database
            try:
                conn = psycopg2.connect(self.db_url)
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
                # Database not available or error, use defaults
                print(f"⚠️ Database error in /api/clients: {e}")

            current_round = last_completed_round
            round_in_progress = False
            submitted_clients = set()
            selected_clients = set()

            # Always fetch fresh server status for real-time updates
            try:
                import requests
                server_status = requests.get('http://server:5000/status', timeout=10).json()
            except Exception as e:
                server_status = None
                print(f"⚠️ Server status error: {e}")

            if server_status:
                round_in_progress = server_status.get('round_in_progress', False)
                if round_in_progress:
                    current_round = server_status.get('current_round', last_completed_round)
                    submitted_clients = set(server_status.get('submitted_clients', []))
                    selected_clients = set(server_status.get('selected_clients', []))
                else:
                    selected_clients = set()
            else:
                # Fallback to database detection if server not available
                # Check if there's an incomplete round in database
                try:
                    conn_fallback = psycopg2.connect(self.db_url)
                    cursor_fallback = conn_fallback.cursor()
                    cursor_fallback.execute("""
                        SELECT round_number, global_test_accuracy
                        FROM round_metrics
                        ORDER BY round_number DESC
                        LIMIT 1
                    """)
                    latest_fallback = cursor_fallback.fetchone()

                    if latest_fallback and latest_fallback[1] is None:
                        round_in_progress = True
                        current_round = latest_fallback[0]
                        # Get submitted clients from database instead of using empty set
                        cursor_fallback.execute("""
                            SELECT DISTINCT client_id
                            FROM client_metrics
                            WHERE round_number = %s
                        """, (current_round,))
                        submitted_clients = set(row[0] for row in cursor_fallback.fetchall())
                        # Assume all expected clients are selected for this round
                        selected_clients = set(range(expected_clients))
                    else:
                        round_in_progress = False
                        submitted_clients = set()
                        selected_clients = set()
                    
                    conn_fallback.close()
                except Exception as e:
                    print(f"⚠️ Fallback detection error: {e}")
                    round_in_progress = False
                    submitted_clients = set()
                    selected_clients = set()

            # Determine client states
            clients = []
            for client_id in range(expected_clients):
                if round_in_progress:
                    if client_id in submitted_clients:
                        state = "active"
                    else:
                        state = "training"
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
        
        @self.app.route('/api/bandwidth')
        def get_bandwidth_data():
            """Get bandwidth usage per client"""
            try:
                conn = psycopg2.connect(self.db_url)
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

        @self.app.route('/api/latency')
        def get_latency_data():
            """Get latency data for plotting"""
            try:
                conn = psycopg2.connect(self.db_url)
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

        @self.app.route('/api/energy')
        def get_energy_data():
            """Get energy consumption data for plotting"""
            try:
                conn = psycopg2.connect(self.db_url)
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

        @self.app.route('/api/network_metrics')
        def get_network_metrics_data():
            """Get network metrics (packet loss and jitter) for plotting"""
            try:
                conn = psycopg2.connect(self.db_url)
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
                packet_loss = [round(row[1] * 100, 4) if row[1] else None for row in data]  # Convert to percentage
                jitter = [round(row[2], 2) if row[2] else None for row in data]

                return jsonify({
                    "rounds": rounds,
                    "packet_loss": packet_loss,
                    "jitter": jitter
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/rounds_history')
        def get_rounds_history():
            """Get detailed history of all rounds"""
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT round_number,
                           global_test_accuracy,
                           global_test_loss,
                           aggregation_time_sec,
                           num_clients
                    FROM round_metrics
                    ORDER BY round_number DESC
                    LIMIT 20
                """)

                data = cursor.fetchall()
                conn.close()

                rounds = []
                for row in data:
                    rounds.append({
                        "round": row[0],
                        "accuracy": round(row[1], 2) if row[1] else None,
                        "loss": round(row[2], 4) if row[2] else None,
                        "agg_time": round(row[3], 2) if row[3] else None,
                        "clients": row[4]
                    })

                return jsonify({"rounds": rounds})

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/start_round', methods=['POST'])
        def start_round():
            """Proxy to start a round on the FL server"""
            try:
                import requests
                response = requests.post('http://server:5000/start_round', timeout=5)
                return jsonify(response.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/xfl/set_strategy', methods=['POST'])
        def set_xfl_strategy():
            """
            Proxy to set XFL strategy on FL server
            
            POST body:
            {
                "strategy": "first_n_layers",
                "param": 2
            }
            """
            try:
                import requests
                data = request.get_json()
                
                response = requests.post(
                    'http://server:5000/xfl/set_strategy',
                    json=data,
                    timeout=5
                )
                return jsonify(response.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/export')
        def export_csv():
            """Export metrics to CSV"""
            try:
                conn = psycopg2.connect(self.db_url)

                import pandas as pd
                df_rounds = pd.read_sql_query("SELECT * FROM round_metrics", conn)
                df_clients = pd.read_sql_query("SELECT * FROM client_metrics", conn)

                conn.close()

                df_rounds.to_csv('/app/logs/round_metrics.csv', index=False)
                df_clients.to_csv('/app/logs/client_metrics.csv', index=False)

                return jsonify({"status": "success", "message": "Data exported to logs/ folder"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def run(self, debug: bool = False):
        """
        Run dashboard server
        
        Args:
            debug: Enable debug mode
        """
        print(f"\n🚀 Starting Dashboard Server on http://localhost:{self.port}")
        print(f"   Open your browser and navigate to: http://localhost:{self.port}")
        print(f"   Press Ctrl+C to stop\n")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


def run_dashboard(db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics", port: int = 5001):
    """
    Run dashboard server

    Args:
        db_url: PostgreSQL database URL
        port: Port number
    """
    dashboard = DashboardServer(db_url=db_url, port=port)
    dashboard.run(debug=False)


if __name__ == "__main__":
    """Run dashboard server"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config_parser import load_config

    # Load configuration to get database URL
    try:
        config = load_config()
        db_url = config.server.metrics_db_url
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        print("   Using default PostgreSQL URL...")
        db_url = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics"

    run_dashboard(db_url=db_url, port=5001)
