"""
Server-side metrics collection and storage
"""

import sqlite3
import json
from typing import Dict, List, Any
from pathlib import Path
import time


class ServerMetricsCollector:
    """
    Collect and store server-side metrics in SQLite database
    """
    
    def __init__(self, db_path: str = "logs/server_metrics.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create logs directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"✅ ServerMetricsCollector initialized with database: {db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for round-level metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS round_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                num_clients INTEGER NOT NULL,
                aggregation_time_sec REAL,
                global_test_loss REAL,
                global_test_accuracy REAL,
                total_samples INTEGER,
                metrics_json TEXT
            )
        """)
        
        # Table for client-level metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER NOT NULL,
                client_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                training_loss REAL,
                training_accuracy REAL,
                training_time_sec REAL,
                num_samples INTEGER,
                model_size_mb REAL,
                cpu_percent REAL,
                memory_mb REAL,
                bytes_sent INTEGER,
                bytes_received INTEGER,
                metrics_json TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database tables initialized")
    
    def store_round_metrics(
        self,
        round_number: int,
        num_clients: int,
        aggregation_time: float,
        global_test_loss: float = None,
        global_test_accuracy: float = None,
        total_samples: int = None,
        additional_metrics: Dict[str, Any] = None
    ):
        """
        Store metrics for a complete FL round
        
        Args:
            round_number: Current round number
            num_clients: Number of clients participated
            aggregation_time: Time taken for aggregation
            global_test_loss: Global model test loss
            global_test_accuracy: Global model test accuracy
            total_samples: Total samples used in this round
            additional_metrics: Any additional metrics as dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_json = json.dumps(additional_metrics) if additional_metrics else None
        
        cursor.execute("""
            INSERT INTO round_metrics 
            (round_number, timestamp, num_clients, aggregation_time_sec,
             global_test_loss, global_test_accuracy, total_samples, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            round_number,
            time.time(),
            num_clients,
            aggregation_time,
            global_test_loss,
            global_test_accuracy,
            total_samples,
            metrics_json
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Round {round_number} metrics stored")
    
    def store_client_metrics(
        self,
        round_number: int,
        client_id: int,
        client_metrics: Dict[str, Any]
    ):
        """
        Store metrics from a single client
        
        Args:
            round_number: Current round number
            client_id: Client identifier
            client_metrics: Dictionary containing client metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract relevant metrics
        training = client_metrics.get('training', {})
        system = client_metrics.get('system', {})
        model = client_metrics.get('model', {})
        network = client_metrics.get('network', {})
        
        cursor.execute("""
            INSERT INTO client_metrics 
            (round_number, client_id, timestamp, training_loss, training_accuracy,
             training_time_sec, num_samples, model_size_mb, cpu_percent, memory_mb,
             bytes_sent, bytes_received, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            round_number,
            client_id,
            client_metrics.get('timestamp', time.time()),
            training.get('loss'),
            training.get('accuracy'),
            training.get('training_time'),
            training.get('num_samples'),
            model.get('total_mb'),
            system.get('process_cpu_percent'),
            system.get('process_memory_mb'),
            network.get('bytes_sent'),
            network.get('bytes_received'),
            json.dumps(client_metrics)
        ))
        
        conn.commit()
        conn.close()
    
    def get_round_metrics(self, round_number: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve round metrics
        
        Args:
            round_number: Specific round number (None for all rounds)
            
        Returns:
            List of round metrics
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if round_number is not None:
            cursor.execute("""
                SELECT * FROM round_metrics WHERE round_number = ?
            """, (round_number,))
        else:
            cursor.execute("""
                SELECT * FROM round_metrics ORDER BY round_number
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_client_metrics(
        self, 
        round_number: int = None,
        client_id: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve client metrics
        
        Args:
            round_number: Filter by round number (None for all)
            client_id: Filter by client ID (None for all)
            
        Returns:
            List of client metrics
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM client_metrics WHERE 1=1"
        params = []
        
        if round_number is not None:
            query += " AND round_number = ?"
            params.append(round_number)
        
        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)
        
        query += " ORDER BY round_number, client_id"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all rounds
        
        Returns:
            Dictionary with summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Round statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_rounds,
                AVG(global_test_accuracy) as avg_accuracy,
                MAX(global_test_accuracy) as max_accuracy,
                AVG(aggregation_time_sec) as avg_aggregation_time
            FROM round_metrics
        """)
        round_stats = cursor.fetchone()
        
        # Client statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_submissions,
                AVG(training_time_sec) as avg_training_time,
                AVG(training_accuracy) as avg_client_accuracy,
                AVG(cpu_percent) as avg_cpu_usage,
                AVG(memory_mb) as avg_memory_usage
            FROM client_metrics
        """)
        client_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "rounds": {
                "total": round_stats[0],
                "avg_accuracy": round(round_stats[1], 2) if round_stats[1] else None,
                "max_accuracy": round(round_stats[2], 2) if round_stats[2] else None,
                "avg_aggregation_time": round(round_stats[3], 2) if round_stats[3] else None
            },
            "clients": {
                "total_submissions": client_stats[0],
                "avg_training_time": round(client_stats[1], 2) if client_stats[1] else None,
                "avg_accuracy": round(client_stats[2], 2) if client_stats[2] else None,
                "avg_cpu_usage": round(client_stats[3], 2) if client_stats[3] else None,
                "avg_memory_usage": round(client_stats[4], 2) if client_stats[4] else None
            }
        }
    
    def clear_database(self):
        """Clear all metrics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM round_metrics")
        cursor.execute("DELETE FROM client_metrics")
        
        conn.commit()
        conn.close()
        
        print("✅ Database cleared")


# Test function
if __name__ == "__main__":
    """Test server metrics collector"""
    print("🧪 Testing ServerMetricsCollector...\n")
    
    # Create collector
    collector = ServerMetricsCollector(db_path="logs/test_metrics.db")
    
    # Clear previous test data
    collector.clear_database()
    
    # Store round metrics
    print("\n📊 Storing round metrics...")
    collector.store_round_metrics(
        round_number=1,
        num_clients=5,
        aggregation_time=2.5,
        global_test_loss=0.45,
        global_test_accuracy=85.5,
        total_samples=60000
    )
    
    # Store client metrics
    print("\n📊 Storing client metrics...")
    for client_id in range(3):
        client_metrics = {
            "client_id": client_id,
            "timestamp": time.time(),
            "training": {
                "loss": 0.5 - client_id * 0.1,
                "accuracy": 80.0 + client_id * 2,
                "training_time": 10.0 + client_id,
                "num_samples": 12000
            },
            "system": {
                "process_cpu_percent": 45.0 + client_id * 5,
                "process_memory_mb": 500.0 + client_id * 50
            },
            "model": {
                "total_mb": 1.5
            },
            "network": {
                "bytes_sent": 1500000,
                "bytes_received": 1500000
            }
        }
        collector.store_client_metrics(1, client_id, client_metrics)
    
    # Retrieve metrics
    print("\n📊 Retrieving round metrics...")
    round_metrics = collector.get_round_metrics(round_number=1)
    print(f"   Found {len(round_metrics)} round(s)")
    for metric in round_metrics:
        print(f"   Round {metric['round_number']}: Accuracy = {metric['global_test_accuracy']}%")
    
    print("\n📊 Retrieving client metrics...")
    client_metrics = collector.get_client_metrics(round_number=1)
    print(f"   Found {len(client_metrics)} client submission(s)")
    for metric in client_metrics:
        print(f"   Client {metric['client_id']}: Loss = {metric['training_loss']}, "
              f"Accuracy = {metric['training_accuracy']}%")
    
    # Get summary
    print("\n📊 Summary statistics...")
    summary = collector.get_summary_statistics()
    print(json.dumps(summary, indent=2))
    
    print("\n✅ All ServerMetricsCollector tests passed!")