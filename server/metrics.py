"""
Server-side metrics collection and storage
"""

import psycopg2
import psycopg2.extras
import json
from typing import Dict, List, Any
from pathlib import Path
import time


class ServerMetricsCollector:
    """
    Collect and store server-side metrics in PostgreSQL database
    """
    
    def __init__(self, db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics"):
        """
        Args:
            db_url: PostgreSQL database URL
        """
        self.db_url = db_url

        # Initialize database
        self._init_database()

        print(f"✅ ServerMetricsCollector initialized with database: {db_url}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        # Table for round-level metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS round_metrics (
                id SERIAL PRIMARY KEY,
                round_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                num_clients INTEGER NOT NULL,
                aggregation_time_sec REAL,
                global_test_loss REAL,
                global_test_accuracy REAL,
                total_samples INTEGER,
                metrics_json JSONB
            )
        """)

        # Table for client-level metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_metrics (
                id SERIAL PRIMARY KEY,
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
                latency_ms REAL,
                packet_loss_rate REAL,
                jitter_ms REAL,
                energy_joules REAL,
                energy_wh REAL,
                avg_power_watts REAL,
                metrics_json JSONB
            )
        """)

        # Add missing columns if they don't exist (for schema migration)
        cursor.execute("ALTER TABLE client_metrics ADD COLUMN IF NOT EXISTS energy_joules REAL")
        cursor.execute("ALTER TABLE client_metrics ADD COLUMN IF NOT EXISTS energy_wh REAL")
        cursor.execute("ALTER TABLE client_metrics ADD COLUMN IF NOT EXISTS avg_power_watts REAL")

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
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO round_metrics
            (round_number, timestamp, num_clients, aggregation_time_sec,
             global_test_loss, global_test_accuracy, total_samples, metrics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            round_number,
            time.time(),
            num_clients,
            aggregation_time,
            global_test_loss,
            global_test_accuracy,
            total_samples,
            psycopg2.extras.Json(additional_metrics) if additional_metrics else None
        ))

        conn.commit()
        conn.close()

        print(f"✅ Round {round_number} metrics stored")

    def update_round_metrics_with_evaluation(
        self,
        round_number: int,
        global_test_loss: float,
        global_test_accuracy: float
    ):
        """
        Update round metrics with evaluation results (loss and accuracy)

        Args:
            round_number: Round number to update
            global_test_loss: Global model test loss
            global_test_accuracy: Global model test accuracy
        """
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE round_metrics
            SET global_test_loss = %s, global_test_accuracy = %s
            WHERE round_number = %s
        """, (
            global_test_loss,
            global_test_accuracy,
            round_number
        ))

        conn.commit()
        conn.close()

        print(f"✅ Round {round_number} metrics updated with evaluation results")
    
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
        conn = psycopg2.connect(self.db_url)
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
             bytes_sent, bytes_received, latency_ms, packet_loss_rate, jitter_ms,
             energy_joules, energy_wh, avg_power_watts, metrics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            network.get('latency_ms'),
            network.get('packet_loss_rate'),
            network.get('jitter_ms'),
            client_metrics.get('energy', {}).get('energy_joules'),
            client_metrics.get('energy', {}).get('energy_wh'),
            client_metrics.get('energy', {}).get('avg_power_watts'),
            psycopg2.extras.Json(client_metrics)
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
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if round_number is not None:
            cursor.execute("""
                SELECT * FROM round_metrics WHERE round_number = %s
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
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = "SELECT * FROM client_metrics WHERE 1=1"
        params = []

        if round_number is not None:
            query += " AND round_number = %s"
            params.append(round_number)

        if client_id is not None:
            query += " AND client_id = %s"
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
        conn = psycopg2.connect(self.db_url)
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
                AVG(memory_mb) as avg_memory_usage,
                AVG(latency_ms) as avg_latency,
                AVG(packet_loss_rate) as avg_packet_loss,
                AVG(jitter_ms) as avg_jitter,
                AVG(energy_wh) as avg_energy_wh
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
                "avg_memory_usage": round(client_stats[4], 2) if client_stats[4] else None,
                "avg_latency_ms": round(client_stats[5], 2) if client_stats[5] else None,
                "avg_packet_loss_rate": round(client_stats[6], 4) if client_stats[6] else None,
                "avg_jitter_ms": round(client_stats[7], 2) if client_stats[7] else None,
                "avg_energy_wh": round(client_stats[8], 4) if client_stats[8] else None
            }
        }
    
    def clear_database(self):
        """Clear all metrics from database"""
        conn = psycopg2.connect(self.db_url)
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
    collector = ServerMetricsCollector(db_url="postgresql://postgres:newpassword@localhost:5432/xfl_metrics")

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
