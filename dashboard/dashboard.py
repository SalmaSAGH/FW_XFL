"""
Real-time dashboard for monitoring FL experiments with XFL support
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
import json
from pathlib import Path
import threading
import time
import os


class DashboardServer:
    """
    Real-time dashboard server for FL monitoring with XFL
    """
    
    def __init__(self, db_path: str = "logs/server_metrics.db", port: int = 5001):
        """
        Args:
            db_path: Path to metrics database
            port: Port for dashboard server
        """
        self.db_path = db_path
        self.port = port
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))
        CORS(self.app)
        
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
                expected_clients = int(os.getenv('NUM_CLIENTS', '5'))
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT MAX(round_number) FROM round_metrics")
                max_completed_round = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT COUNT(DISTINCT client_id) 
                    FROM client_metrics
                    WHERE round_number = ?
                """, (max_completed_round,))
                active_clients_last_round = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT round_number, global_test_accuracy, global_test_loss
                    FROM round_metrics
                    ORDER BY round_number DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()
                
                round_in_progress = False
                submissions_received = 0
                current_round = max_completed_round
                xfl_strategy = "all_layers"
                xfl_param = 3
                
                try:
                    import requests
                    server_status = requests.get('http://server:5000/status', timeout=1).json()
                    round_in_progress = server_status.get('round_in_progress', False)
                    submissions_received = server_status.get('submissions_received', 0)
                    server_current_round = server_status.get('current_round', 0)
                    xfl_strategy = server_status.get('xfl_strategy', 'all_layers')
                    xfl_param = server_status.get('xfl_param', 3)
                    num_layers = server_status.get('num_layers', 0)

                    if server_current_round > 0:
                        current_round = server_current_round
                except:
                    num_layers = 0
                    pass
                
                conn.close()
                
                return jsonify({
                    "current_round": current_round,
                    "total_rounds": max_completed_round,
                    "total_clients": active_clients_last_round if active_clients_last_round > 0 else expected_clients,
                    "latest_accuracy": round(latest[1], 2) if latest and latest[1] else None,
                    "latest_loss": round(latest[2], 4) if latest and latest[2] else None,
                    "round_in_progress": round_in_progress,
                    "clients_expected": expected_clients,
                    "submissions_received": submissions_received,
                    "xfl_strategy": xfl_strategy,
                    "xfl_param": xfl_param,
                    "num_layers": num_layers
                })
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/accuracy')
        def get_accuracy_data():
            """Get accuracy data for plotting"""
            try:
                conn = sqlite3.connect(self.db_path)
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
                conn = sqlite3.connect(self.db_path)
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
            try:
                expected_clients = int(os.getenv('NUM_CLIENTS', '5'))
                
                conn = sqlite3.connect(self.db_path)
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
                
                current_round = last_completed_round
                training_client_ids = set()
                
                try:
                    import requests
                    server_status = requests.get('http://server:5000/status', timeout=1).json()
                    if server_status.get('round_in_progress', False):
                        current_round = server_status.get('current_round', last_completed_round)
                        
                        cursor.execute("""
                            SELECT DISTINCT client_id
                            FROM client_metrics
                            WHERE round_number = ?
                        """, (current_round,))
                        submitted_clients = set(row[0] for row in cursor.fetchall())
                        
                        all_expected = set(range(expected_clients))
                        training_client_ids = all_expected - submitted_clients
                except:
                    pass
                
                cursor.execute("""
                    SELECT DISTINCT client_id
                    FROM client_metrics
                    WHERE round_number = ?
                """, (last_completed_round,))
                active_in_last_round = set(row[0] for row in cursor.fetchall())
                
                conn.close()
                
                clients = []
                for client_id in range(expected_clients):
                    if client_id in client_data_map:
                        row = client_data_map[client_id]
                        
                        if client_id in training_client_ids:
                            state = "training"
                        elif client_id in active_in_last_round:
                            state = "active"
                        else:
                            state = "idle"
                        
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
                        state = "training" if client_id in training_client_ids else "idle"
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
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/bandwidth')
        def get_bandwidth_data():
            """Get bandwidth usage per client"""
            try:
                conn = sqlite3.connect(self.db_path)
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
        
        @self.app.route('/api/rounds_history')
        def get_rounds_history():
            """Get detailed history of all rounds"""
            try:
                conn = sqlite3.connect(self.db_path)
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
                conn = sqlite3.connect(self.db_path)
                
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


def run_dashboard(db_path: str = "logs/server_metrics.db", port: int = 5001):
    """
    Run dashboard server
    
    Args:
        db_path: Path to metrics database
        port: Port number
    """
    dashboard = DashboardServer(db_path=db_path, port=port)
    dashboard.run(debug=False)


if __name__ == "__main__":
    """Run dashboard server"""
    import sys
    
    db_path = "logs/server_metrics.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    if not Path(db_path).exists():
        print(f"❌ Error: Database not found at {db_path}")
        print(f"   Please run an experiment first")
        sys.exit(1)
    
    run_dashboard(db_path=db_path, port=5001)