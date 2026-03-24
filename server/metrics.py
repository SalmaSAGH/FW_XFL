"""
Server-side metrics collection and storage
"""

import psycopg2
import psycopg2.extras
import psycopg2.pool
import json
from typing import Dict, List, Any
from pathlib import Path
import time
import threading
import signal
import sys


# Maximum retries for database operations
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # seconds


class ServerMetricsCollector:
    """
    Collect and store server-side metrics in PostgreSQL database
    Uses connection pooling for better performance and resource management
    """
    
    def __init__(self, db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics"):
        """
        Args:
            db_url: PostgreSQL database URL
        """
        self.db_url = db_url

        # Initialize connection pool (min 2, max 10 connections)
        # ThreadedConnectionPool is thread-safe
        self._connection_pool = None
        self._pool_lock = threading.Lock()
        
        # Track if we're shutting down
        self._is_shutting_down = False
        
        # Initialize database and connection pool
        self._init_database()
        
        # Initialize connection pool after tables are created
        self._init_connection_pool()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        print(f"✅ ServerMetricsCollector initialized with database: {db_url}")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (ValueError, OSError):
            # Signal handling not available (e.g., on Windows)
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️  Received signal {signum}, initiating graceful shutdown...")
        self._is_shutting_down = True
        # Try to close the connection pool
        self._close_pool()
        sys.exit(0)
    
    def _init_connection_pool(self):
        """Initialize the connection pool with improved settings for long-running applications"""
        try:
            # Parse the database URL to get connection parameters
            # Format: postgresql://user:password@host:port/dbname
            db_url = self.db_url.replace('postgresql://', '')
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
            
            # Create connection pool with keepalive settings for long-running applications
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname,
                connect_timeout=10,
                options='-c statement_timeout=30000'  # 30 second query timeout to prevent hanging
            )
            print(f"✅ Connection pool initialized (min: 2, max: 10)")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize connection pool: {e}")
            print("   Falling back to individual connections")
            self._connection_pool = None
    
    def _get_connection(self):
        """Get a connection from the pool or create a new one"""
        if self._connection_pool:
            try:
                conn = self._connection_pool.getconn()
                # Verify the connection is still valid
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return conn
                except:
                    # Connection is invalid, try to get a new one
                    self._connection_pool.putconn(conn, close=True)
                    conn = self._connection_pool.getconn()
                    return conn
            except Exception as e:
                print(f"⚠️ Warning: Could not get connection from pool: {e}")
        
        # Fallback: create a new connection
        return psycopg2.connect(self.db_url)
    
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        if self._connection_pool and conn:
            try:
                self._connection_pool.putconn(conn)
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
    
    def _close_pool(self):
        """Close all connections in the pool"""
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
                print("✅ Connection pool closed")
            except Exception as e:
                print(f"⚠️ Warning: Could not close connection pool: {e}")
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute a database operation with retry logic.
        
        Args:
            operation: A callable that takes a connection and cursor as arguments
            *args, **kwargs: Additional arguments to pass to the operation
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            if self._is_shutting_down:
                print("⚠️  Shutting down, skipping database operation")
                return None
                
            conn = None
            try:
                conn = self._get_connection()
                # Set autocommit to False for explicit transaction control
                conn.autocommit = False
                
                # Execute the operation
                result = operation(conn, *args, **kwargs)
                
                # Commit the transaction
                conn.commit()
                
                return result
                
            except psycopg2.OperationalError as e:
                # Network error or connection issue - retry
                last_exception = e
                print(f"⚠️  Database operation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    
            except psycopg2.IntegrityError as e:
                # Integrity constraint violation - don't retry, it's a persistent error
                last_exception = e
                print(f"⚠️  Database integrity error: {e}")
                
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                        
                break  # Don't retry integrity errors
                
            except Exception as e:
                # Other errors - retry
                last_exception = e
                print(f"⚠️  Database operation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    
            finally:
                if conn:
                    conn.autocommit = True  # Reset autocommit
                    self._return_connection(conn)
        
        # All retries failed
        print(f"❌ Database operation failed after {MAX_RETRIES} attempts: {last_exception}")
        raise last_exception
    
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
                strategy VARCHAR(255),
                metrics_json JSONB
            )
        """)

        # Add strategy column if it doesn't exist (for migration)
        cursor.execute("ALTER TABLE round_metrics ADD COLUMN IF NOT EXISTS strategy VARCHAR(255)")

        # ── NEW: Add session_id column (migration-safe) ──────────────────────
        # session_id is a UUID generated once per docker-compose up.
        # Every round started in that container lifetime shares the same session_id.
        # This lets the History page group rounds into true "sessions" without
        # relying on round-number gap detection.
        cursor.execute("ALTER TABLE round_metrics ADD COLUMN IF NOT EXISTS session_id VARCHAR(36)")

        # ── NEW: Table that registers each server startup ─────────────────────
        # One row is inserted at docker-compose up; the session_id is reused
        # for every round until docker-compose down.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server_sessions (
                session_id   VARCHAR(36)  PRIMARY KEY,
                started_at   REAL         NOT NULL,
                hostname     VARCHAR(255),
                notes        TEXT
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
        strategy: str = None,
        additional_metrics: Dict[str, Any] = None,
        session_id: str = None          # ← NEW parameter
    ):
        """
        Store metrics for a complete FL round (upsert - updates if exists, inserts if not).
        The session_id tags this round to the current docker-compose up lifetime.
        Uses retry logic for better reliability.
        """
        
        def _do_store(conn):
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE round_metrics
                SET timestamp = %s, num_clients = %s, aggregation_time_sec = %s,
                    global_test_loss = %s, global_test_accuracy = %s, total_samples = %s,
                    strategy = %s, metrics_json = %s, session_id = %s
                WHERE round_number = %s AND session_id = %s
            """, (
                time.time(), num_clients, aggregation_time,
                global_test_loss, global_test_accuracy, total_samples,
                strategy,
                psycopg2.extras.Json(additional_metrics) if additional_metrics else None,
                session_id,
                round_number, session_id          # WHERE clause
            ))
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO round_metrics (
                        round_number, timestamp, num_clients, aggregation_time_sec,
                        global_test_loss, global_test_accuracy, total_samples,
                        strategy, metrics_json, session_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    round_number, time.time(), num_clients, aggregation_time,
                    global_test_loss, global_test_accuracy, total_samples,
                    strategy,
                    psycopg2.extras.Json(additional_metrics) if additional_metrics else None,
                    session_id
                ))
            print(f"✅ Round {round_number} metrics stored/updated | strategy: {strategy} | session: {session_id}")
        
        try:
            self._execute_with_retry(_do_store)
        except Exception as e:
            print(f"❌ Failed to store round metrics after retries: {e}")
            raise

    def update_round_metrics_with_evaluation(
        self,
        round_number: int,
        global_test_loss: float,
        global_test_accuracy: float,
        session_id: str = None          # ← NEW parameter
    ):
        """Update round metrics with evaluation results"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    """UPDATE round_metrics
                       SET global_test_loss = %s, global_test_accuracy = %s
                       WHERE round_number = %s AND session_id = %s""",
                    (global_test_loss, global_test_accuracy, round_number, session_id)
                )
            else:
                # Fallback for callers that don't provide session_id yet
                cursor.execute(
                    """UPDATE round_metrics
                       SET global_test_loss = %s, global_test_accuracy = %s
                       WHERE round_number = %s""",
                    (global_test_loss, global_test_accuracy, round_number)
                )
            conn.commit()
            print(f"✅ Round {round_number} metrics updated with evaluation results")
        finally:
            self._return_connection(conn)
    
    def store_client_metrics(self, round_number: int, client_id: int, client_metrics: Dict[str, Any]):
        """Store metrics from a single client with retry logic"""
        
        def _do_store(conn):
            cursor = conn.cursor()
            training = client_metrics.get('training', {})
            system = client_metrics.get('system', {})
            model = client_metrics.get('model', {})
            network = client_metrics.get('network', {})
            cursor.execute("""
                INSERT INTO client_metrics (round_number, client_id, timestamp, training_loss, training_accuracy,
                 training_time_sec, num_samples, model_size_mb, cpu_percent, memory_mb, bytes_sent, bytes_received,
                 latency_ms, packet_loss_rate, jitter_ms, energy_joules, energy_wh, avg_power_watts, metrics_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (round_number, client_id, client_metrics.get('timestamp', time.time()),
                  training.get('loss'), training.get('accuracy'), training.get('training_time'),
                  training.get('num_samples'), model.get('total_mb'), system.get('process_cpu_percent'),
                  system.get('process_memory_mb'), network.get('bytes_sent'), network.get('bytes_received'),
                  network.get('latency_ms'), network.get('packet_loss_rate'), network.get('jitter_ms'),
                  client_metrics.get('energy', {}).get('energy_joules'),
                  client_metrics.get('energy', {}).get('energy_wh'),
                  client_metrics.get('energy', {}).get('avg_power_watts'),
                  psycopg2.extras.Json(client_metrics)))
        
        try:
            self._execute_with_retry(_do_store)
        except Exception as e:
            print(f"❌ Failed to store client metrics after retries: {e}")
            raise
    
    def get_round_metrics(self, round_number: int = None) -> List[Dict[str, Any]]:
        """Retrieve round metrics"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if round_number is not None:
                cursor.execute("SELECT * FROM round_metrics WHERE round_number = %s", (round_number,))
            else:
                cursor.execute("SELECT * FROM round_metrics ORDER BY round_number")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            self._return_connection(conn)
    
    def get_client_metrics(self, round_number: int = None, client_id: int = None) -> List[Dict[str, Any]]:
        """Retrieve client metrics"""
        conn = self._get_connection()
        try:
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
            return [dict(row) for row in rows]
        finally:
            self._return_connection(conn)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all rounds"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as total_rounds, AVG(global_test_accuracy) as avg_accuracy,
                       MAX(global_test_accuracy) as max_accuracy, AVG(aggregation_time_sec) as avg_aggregation_time
                FROM round_metrics""")
            round_stats = cursor.fetchone()
            cursor.execute("""
                SELECT COUNT(*) as total_submissions, AVG(training_time_sec) as avg_training_time,
                       AVG(training_accuracy) as avg_client_accuracy, AVG(cpu_percent) as avg_cpu_usage,
                       AVG(memory_mb) as avg_memory_usage, AVG(latency_ms) as avg_latency,
                       AVG(packet_loss_rate) as avg_packet_loss, AVG(jitter_ms) as avg_jitter,
                       AVG(energy_wh) as avg_energy_wh FROM client_metrics""")
            client_stats = cursor.fetchone()
            return {
                "rounds": {"total": round_stats[0], "avg_accuracy": round(round_stats[1], 2) if round_stats[1] else None,
                           "max_accuracy": round(round_stats[2], 2) if round_stats[2] else None,
                           "avg_aggregation_time": round(round_stats[3], 2) if round_stats[3] else None},
                "clients": {"total_submissions": client_stats[0], "avg_training_time": round(client_stats[1], 2) if client_stats[1] else None,
                            "avg_accuracy": round(client_stats[2], 2) if client_stats[2] else None,
                            "avg_cpu_usage": round(client_stats[3], 2) if client_stats[3] else None,
                            "avg_memory_usage": round(client_stats[4], 2) if client_stats[4] else None,
                            "avg_latency_ms": round(client_stats[5], 2) if client_stats[5] else None,
                            "avg_packet_loss_rate": round(client_stats[6], 4) if client_stats[6] else None,
                            "avg_jitter_ms": round(client_stats[7], 2) if client_stats[7] else None,
                            "avg_energy_wh": round(client_stats[8], 4) if client_stats[8] else None}
            }
        finally:
            self._return_connection(conn)
    
    def clear_database(self):
        """Clear all metrics from database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM round_metrics")
            cursor.execute("DELETE FROM client_metrics")
            conn.commit()
            print("✅ Database cleared")
        finally:
            self._return_connection(conn)


if __name__ == "__main__":
    print("🧪 Testing ServerMetricsCollector...\n")
    collector = ServerMetricsCollector(db_url="postgresql://postgres:newpassword@localhost:5432/xfl_metrics")
    collector.clear_database()
    print("\n📊 Storing round metrics...")
    collector.store_round_metrics(round_number=1, num_clients=5, aggregation_time=2.5,
                                  global_test_loss=0.45, global_test_accuracy=85.5, total_samples=60000)
    print("\n✅ All ServerMetricsCollector tests passed!")