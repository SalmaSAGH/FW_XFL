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
import requests

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_parser import load_config


class DashboardServer:
    """
    Real-time dashboard server for FL monitoring with XFL
    """

    def __init__(self, db_url: str = "postgresql://postgres:newpassword@postgres:5432/xfl_metrics", port: int = 5001):
        self.db_url = db_url
        self.port = port
        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        CORS(self.app)

        try:
            self.config = load_config()
            print(f"✅ Configuration loaded: {self.config.federated_learning.num_clients} clients, {self.config.federated_learning.num_rounds} rounds")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            self.config = None

        self._server_status_cache = None
        self._cache_timestamp = 0
        self._cache_timeout = 0

        self._setup_routes()

        print(f"✅ Dashboard server initialized on port {port}")
        print(f"   XFL strategy control enabled")

    def _get_num_clients_from_server(self):
        """
        Get numClients from FL server config (source of truth).
        Falls back to NUM_CLIENTS env var, then 40.
        This ensures the dashboard always reflects what the user saved in Config page.
        """
        try:
            resp = requests.get('http://server:5000/api/config', timeout=5)
            cfg = resp.json().get('config', {})
            val = cfg.get('numClients')
            if val is not None:
                return int(val)
        except Exception:
            pass
        # Fallback to environment variable
        return int(os.getenv('NUM_CLIENTS', '40'))

    def _get_active_session_id(self):
        """
        Get the active (latest) session_id from server_sessions.
        Returns None if no sessions exist.
        """
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id FROM server_sessions 
                ORDER BY started_at DESC LIMIT 1
            """)
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"⚠️ Error getting active session: {e}")
            return None

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('dashboard.html')

        @self.app.route('/api/status')
        def get_status():
            try:
                
                num_clients = self._get_num_clients_from_server()

                expected_clients = None
                server_status = None
                try:
                    server_status = requests.get('http://server:5000/api/status', timeout=10).json()
                    expected_clients = server_status.get('clients_expected', None)
                except:
                    pass

                if expected_clients is None:
                    expected_clients = int(os.getenv('NUM_CLIENTS', '40'))

                if self.config:
                    total_rounds_config = self.config.federated_learning.num_rounds
                else:
                    total_rounds_config = int(os.getenv('NUM_ROUNDS', '50'))

                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()

                cursor.execute("""
                               SELECT rm.round_number
                               FROM round_metrics rm
                               INNER JOIN server_sessions ss ON ss.session_id = rm.session_id
                               WHERE rm.global_test_accuracy IS NOT NULL
                               ORDER BY ss.started_at DESC, rm.round_number DESC
                               LIMIT 1
                               """)
                result = cursor.fetchone()
                max_completed_round = result[0] if result else 0

                cursor.execute("""
                               SELECT rm.round_number, rm.global_test_accuracy, rm.global_test_loss, rm.num_clients
                               FROM round_metrics rm
                               INNER JOIN server_sessions ss ON ss.session_id = rm.session_id
                               WHERE rm.global_test_accuracy IS NOT NULL
                               ORDER BY ss.started_at DESC, rm.round_number DESC
                               LIMIT 1
                               """)
                latest = cursor.fetchone()

                if latest and latest[3] is not None and latest[3] > 0:
                    active_clients_last_round = latest[3]
                elif latest:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT client_id)
                        FROM client_metrics
                        WHERE round_number = %s
                    """, (latest[0],))
                    active_clients_last_round = cursor.fetchone()[0] or 0
                else:
                    cursor.execute("""
                        SELECT round_number, num_clients
                        FROM round_metrics
                        WHERE num_clients IS NOT NULL AND num_clients > 0
                        ORDER BY round_number DESC
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    active_clients_last_round = result[1] if result else 0

                round_in_progress = False
                current_round = max_completed_round
                submissions_received = 0

                if latest and latest[1] is None:
                    round_in_progress = True
                    current_round = latest[0]
                    cursor.execute("SELECT DISTINCT client_id FROM client_metrics WHERE round_number = %s", (current_round,))
                    submissions_received = len(cursor.fetchall())

                xfl_strategy = "all_layers"
                xfl_param = 3
                num_layers = 0
                submitted_clients = []
                selected_clients = []
                network_latency = 0
                network_bandwidth = 10
                network_packet_loss = 0
                cpu_limit = 100
                ram_limit = 512
                active_clients_percent = 100
                data_distribution = 'iid'

                if server_status:
                    round_in_progress = server_status.get('round_in_progress', round_in_progress)
                    current_round = server_status.get('current_round', current_round)
                    submissions_received = server_status.get('submissions_received', submissions_received)
                    xfl_strategy = server_status.get('xfl_strategy', 'all_layers')
                    xfl_param = server_status.get('xfl_param', 3)
                    num_layers = server_status.get('num_layers', 0)
                    submitted_clients = server_status.get('submitted_clients', [])
                    selected_clients = server_status.get('selected_clients', [])
                    network_latency = server_status.get('network_latency', 0)
                    network_bandwidth = server_status.get('network_bandwidth', 10)
                    network_packet_loss = server_status.get('network_packet_loss', 0)
                    cpu_limit = server_status.get('cpu_limit', 100)
                    ram_limit = server_status.get('ram_limit', 512)
                    active_clients_percent = server_status.get('active_clients_percent', 100)
                    data_distribution = server_status.get('data_distribution', 'iid')

                conn.close()

                return jsonify({
                    "current_round": current_round,
                    "total_rounds": total_rounds_config,
                    # ── FIX: always use numClients from server config ──────────
                    "total_clients": num_clients,
                    "latest_accuracy": round(latest[1], 2) if latest and latest[1] else None,
                    "latest_loss": round(latest[2], 4) if latest and latest[2] else None,
                    "round_in_progress": round_in_progress,
                    "clients_expected": expected_clients,
                    "submissions_received": submissions_received,
                    "xfl_strategy": xfl_strategy,
                    "xfl_param": xfl_param,
                    "num_layers": num_layers,
                    "submitted_clients": submitted_clients,
                    "selected_clients": selected_clients,
                    "network_latency": network_latency,
                    "network_bandwidth": network_bandwidth,
                    "network_packet_loss": network_packet_loss,
                    "cpu_limit": cpu_limit,
                    "ram_limit": ram_limit,
                    "active_clients_percent": active_clients_percent,
                    "data_distribution": data_distribution,
                })

            except Exception as e:
                import traceback
                print(f"ERROR in get_status: {traceback.format_exc()}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/accuracy')
        def get_accuracy_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT round_number, global_test_accuracy
                    FROM round_metrics ORDER BY round_number
                """)
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "accuracy": [round(row[1], 2) if row[1] is not None else None for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/loss')
        def get_loss_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT round_number, global_test_loss
                    FROM round_metrics ORDER BY round_number
                """)
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "loss": [round(row[1], 4) if row[1] is not None else None for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/clients')
        def get_clients_data():
            # ── FIX: get numClients from server config instead of env var ─────
            # This makes the Pi grid reflect whatever the user saved in Config.
            expected_clients = self._get_num_clients_from_server()

            client_data_map = {}
            last_completed_round = 0
            active_in_last_round = set()

            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id,
                           AVG(training_accuracy), AVG(training_time_sec),
                           AVG(cpu_percent), AVG(memory_mb),
                           MAX(round_number), COUNT(*)
                    FROM client_metrics
                    GROUP BY client_id ORDER BY client_id
                """)
                client_data_map = {row[0]: row for row in cursor.fetchall()}

                cursor.execute("""
                               SELECT rm.round_number, rm.metrics_json
                               FROM round_metrics rm
                               INNER JOIN server_sessions ss ON ss.session_id = rm.session_id
                               WHERE rm.global_test_accuracy IS NOT NULL
                               ORDER BY ss.started_at DESC, rm.round_number DESC
                               LIMIT 1
                               """)
                row = cursor.fetchone()
                if row:
                    last_completed_round = row[0]
                    metrics_json = row[1]
                    if metrics_json and 'submitted_clients' in metrics_json:
                        active_in_last_round = set(metrics_json['submitted_clients'])
                    else:
                        cursor.execute("""
                                       SELECT DISTINCT client_id FROM client_metrics
                                       WHERE round_number = %s
                                       """, (last_completed_round,))
                        active_in_last_round = set(r[0] for r in cursor.fetchall())
                else:
                    last_completed_round = 0
                    row = cursor.fetchone()
                    print(f"[DEBUG] last_completed_round={last_completed_round}, metrics_json={row[0] if row else None}")
                    if row and row[0] and 'submitted_clients' in row[0]:
                        active_in_last_round = set(row[0]['submitted_clients'])
                    
                    else:
                        cursor.execute("""
                                       SELECT DISTINCT client_id FROM client_metrics
                                       WHERE round_number = %s
                                       """, (last_completed_round,))
                        active_in_last_round = set(r[0] for r in cursor.fetchall())
                        print(f"[DEBUG] active_in_last_round from client_metrics: {active_in_last_round}")
                conn.close()
            except Exception as e:
                print(f"⚠️ Database error in /api/clients: {e}")

            current_round = last_completed_round
            round_in_progress = False
            submitted_clients = set()
            selected_clients = set()

            try:
                server_status = requests.get('http://server:5000/api/status', timeout=10).json()
                round_in_progress = server_status.get('round_in_progress', False)
                if round_in_progress:
                    current_round = server_status.get('current_round', last_completed_round)
                    submitted_clients = set(server_status.get('submitted_clients', []))
                    selected_clients = set(server_status.get('selected_clients', []))
            except Exception as e:
                print(f"⚠️ Server status error: {e}")

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
                    state = "active" if client_id in active_in_last_round else "idle"

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
                        "avg_accuracy": 0, "avg_time": 0,
                        "avg_cpu": 0, "avg_memory": 0,
                        "num_rounds": 0, "state": state
                    })

            return jsonify({"clients": clients})

        @self.app.route('/api/bandwidth')
        def get_bandwidth_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                session_id = self._get_active_session_id()
                if session_id:
                    # FIX: Show metrics for all rounds with client submissions, not just evaluated rounds
                    cursor.execute("""
                        SELECT cm.round_number, AVG(cm.bytes_sent / 1048576.0) as avg_bw_mb
                        FROM client_metrics cm 
                        INNER JOIN round_metrics rm ON cm.round_number = rm.round_number 
                        WHERE rm.session_id = %s
                        GROUP BY cm.round_number 
                        ORDER BY cm.round_number
                    """, (session_id,))
                else:
                    cursor.execute("""
                        SELECT round_number, 0.0 as avg_bw_mb 
                        FROM round_metrics 
                        WHERE session_id IS NULL 
                        LIMIT 1
                    """)
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "bandwidth_mb": [round(row[1], 2) if row[1] else 0.0 for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/latency')
        def get_latency_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                session_id = self._get_active_session_id()
                if session_id:
                    # FIX: Show metrics for all rounds with client submissions
                    cursor.execute("""
                        SELECT cm.round_number, AVG(cm.latency_ms)
                        FROM client_metrics cm 
                        INNER JOIN round_metrics rm ON cm.round_number = rm.round_number 
                        WHERE rm.session_id = %s
                        GROUP BY cm.round_number 
                        ORDER BY cm.round_number
                    """, (session_id,))
                else:
                    cursor.execute("SELECT NULL, NULL")
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "latency": [round(row[1], 2) if row[1] is not None else None for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/energy')
        def get_energy_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                session_id = self._get_active_session_id()
                if session_id:
                    # FIX: Show metrics for all rounds with client submissions
                    cursor.execute("""
                        SELECT cm.round_number, AVG(cm.energy_wh)
                        FROM client_metrics cm 
                        INNER JOIN round_metrics rm ON cm.round_number = rm.round_number 
                        WHERE rm.session_id = %s
                        GROUP BY cm.round_number 
                        ORDER BY cm.round_number
                    """, (session_id,))
                else:
                    cursor.execute("SELECT NULL, NULL")
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "energy": [round(row[1], 4) if row[1] is not None else None for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/network_metrics')
        def get_network_metrics_data():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                session_id = self._get_active_session_id()
                if session_id:
                    # FIX: Show metrics for all rounds with client submissions
                    cursor.execute("""
                        SELECT cm.round_number,
                               COALESCE(AVG(cm.packet_loss_rate), 0.0) as avg_packet_loss,
                               COALESCE(AVG(cm.jitter_ms), 0.0) as avg_jitter
                        FROM client_metrics cm 
                        INNER JOIN round_metrics rm ON cm.round_number = rm.round_number 
                        WHERE rm.session_id = %s
                        GROUP BY cm.round_number 
                        ORDER BY cm.round_number
                    """, (session_id,))
                else:
                    cursor.execute("SELECT NULL, NULL, NULL")
                data = cursor.fetchall()
                conn.close()
                return jsonify({
                    "rounds": [row[0] for row in data],
                    "packet_loss": [round(row[1] * 100, 4) if row[1] is not None else None for row in data],
                    "jitter": [round(row[2], 2) if row[2] is not None else None for row in data]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/rounds_history')
        def get_rounds_history():
            try:
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT round_number, global_test_accuracy, global_test_loss,
                           aggregation_time_sec, num_clients, strategy
                    FROM round_metrics
                    WHERE global_test_accuracy IS NOT NULL
                    ORDER BY round_number DESC LIMIT 50
                """)
                data = cursor.fetchall()
                conn.close()
                return jsonify({"rounds": [{
                    "round": row[0],
                    "accuracy": round(row[1], 2) if row[1] else None,
                    "loss": round(row[2], 4) if row[2] else None,
                    "agg_time": round(row[3], 2) if row[3] else None,
                    "clients": row[4],
                    "strategy": row[5] if row[5] else 'all_layers'
                } for row in data]})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # ── Proxy routes to FL server ─────────────────────────────────────────

        @self.app.route('/api/register', methods=['POST'])
        def register():
            try:
                response = requests.post('http://server:5000/api/register', json=request.get_json(), timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/login', methods=['POST'])
        def login():
            try:
                response = requests.post('http://server:5000/api/login', json=request.get_json(), timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/logout', methods=['POST'])
        def logout():
            try:
                response = requests.post('http://server:5000/api/logout', json=request.get_json(), timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/verify_token', methods=['POST'])
        def verify_token():
            try:
                response = requests.post('http://server:5000/api/verify_token', json=request.get_json(), timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            try:
                response = requests.get('http://server:5000/api/config', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/config/save', methods=['POST'])
        def save_config():
            try:
                response = requests.post('http://server:5000/api/config/save', json=request.get_json(), timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/start_round', methods=['POST'])
        def start_round():
            try:
                response = requests.post('http://server:5000/api/start_round', timeout=5)
                return jsonify(response.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/xfl/set_strategy', methods=['POST'])
        def set_xfl_strategy():
            try:
                response = requests.post('http://server:5000/xfl/set_strategy', json=request.get_json(), timeout=5)
                return jsonify(response.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/metrics/reset', methods=['POST'])
        def reset_metrics():
            try:
                response = requests.post('http://server:5000/api/metrics/reset', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/export')
        def export_csv():
            try:
                conn = psycopg2.connect(self.db_url)
                import pandas as pd
                pd.read_sql_query("SELECT * FROM round_metrics", conn).to_csv('/app/logs/round_metrics.csv', index=False)
                pd.read_sql_query("SELECT * FROM client_metrics", conn).to_csv('/app/logs/client_metrics.csv', index=False)
                conn.close()
                return jsonify({"status": "success", "message": "Data exported to logs/ folder"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/sessions', methods=['GET'])
        def dse_sessions():
            try:
                response = requests.get('http://server:5000/api/dse/sessions', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/results/<session_id>', methods=['GET'])
        def dse_results(session_id):
            try:
                response = requests.get(f'http://server:5000/api/dse/results/{session_id}', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/sweep', methods=['POST'])
        def dse_sweep():
            try:
                response = requests.post('http://server:5000/api/dse/sweep', json=request.get_json(), timeout=600)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/status/<session_id>', methods=['GET'])
        def dse_status(session_id):
            try:
                response = requests.get(f'http://server:5000/api/dse/status/{session_id}', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/progress/<session_id>', methods=['GET'])
        def dse_progress(session_id):
            try:
                response = requests.get(f'http://server:5000/api/dse/progress/{session_id}', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/all_results', methods=['GET'])
        def dse_all_results():
            try:
                response = requests.get('http://server:5000/api/dse/all_results', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/dse/reset', methods=['POST'])
        def dse_reset():
            try:
                response = requests.post('http://server:5000/api/dse/reset', timeout=10)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/history_by_strategy')
        def get_history_by_strategy():
            try:
                try:
                    response = requests.get('http://server:5000/api/history_by_strategy', timeout=30)
                except:
                    response = requests.get('http://localhost:5000/api/history_by_strategy', timeout=30)
                return jsonify(response.json()), response.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, debug: bool = False):
        print(f"\n🚀 Starting Dashboard Server on http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


def run_dashboard(db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics", port: int = 5001):
    DashboardServer(db_url=db_url, port=port).run(debug=False)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config_parser import load_config
    try:
        config = load_config()
        db_url = config.server.metrics_db_url
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        db_url = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics"
    run_dashboard(db_url=db_url, port=5001)

