"""
Server-side management for real Raspberry Pi clients
Adds endpoints and logic to handle physical Raspberry Pi clients
"""

import os
import time
import threading
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import psutil


@dataclass
class PhysicalClient:
    """Represents a real Raspberry Pi client"""
    client_id: int
    ip_address: str
    hostname: str
    username: str = "pi"
    status: str = "disconnected"  # connected, training, idle, error
    last_seen: float = field(default_factory=time.time)
    round_number: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    connection_quality: Dict[str, float] = field(default_factory=lambda: {
        'latency_ms': 0.0,
        'packet_loss_rate': 0.0,
        'jitter_ms': 0.0
    })


class PhysicalClientManager:
    """
    Manages real Raspberry Pi clients connected to the server
    """
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.clients: Dict[int, PhysicalClient] = {}
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def register_client(
        self,
        client_id: int,
        ip_address: str,
        hostname: str = "",
        username: str = "pi"
    ) -> bool:
        """Register a new physical client"""
        with self._lock:
            if client_id in self.clients:
                # Update existing client
                self.clients[client_id].ip_address = ip_address
                self.clients[client_id].hostname = hostname
                self.clients[client_id].last_seen = time.time()
                self.clients[client_id].status = "connected"
                return True
            
            # Create new client
            self.clients[client_id] = PhysicalClient(
                client_id=client_id,
                ip_address=ip_address,
                hostname=hostname or f"raspberry-pi-{client_id}",
                username=username,
                status="connected"
            )
            print(f"✅ Registered physical client {client_id} at {ip_address}")
            return True
    
    def unregister_client(self, client_id: int) -> bool:
        """Unregister a physical client"""
        with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                print(f"🗑️ Unregistered physical client {client_id}")
                return True
            return False
    
    def update_client_status(
        self,
        client_id: int,
        status: str,
        round_number: int = 0
    ) -> bool:
        """Update client status"""
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id].status = status
                self.clients[client_id].last_seen = time.time()
                if round_number > 0:
                    self.clients[client_id].round_number = round_number
                return True
            return False
    
    def update_client_metrics(
        self,
        client_id: int,
        metrics: Dict[str, Any]
    ) -> bool:
        """Update client metrics"""
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id].metrics = metrics
                self.clients[client_id].last_seen = time.time()
                return True
            return False
    
    def measure_connection_quality(self, client_id: int) -> Dict[str, float]:
        """Measure connection quality to a client using ping"""
        with self._lock:
            if client_id not in self.clients:
                return {'latency_ms': 0, 'packet_loss_rate': 1.0, 'jitter_ms': 0}
            
            client = self.clients[client_id]
        
        # Use ping to measure latency
        try:
            import subprocess
            result = subprocess.run(
                ['ping', '-n', '4', client.ip_address],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse Windows ping output
            import re
            stats_match = re.search(r'Average = (\d+)ms', result.stdout)
            
            if stats_match:
                latency_ms = float(stats_match.group(1))
                
                # Estimate packet loss from ping results
                loss_match = re.search(r'(\d+)% loss', result.stdout)
                packet_loss = float(loss_match.group(1)) / 100.0 if loss_match else 0.0
                
                # Calculate jitter (simplified)
                jitter_ms = latency_ms * 0.1  # Approximate
                
                with self._lock:
                    client.connection_quality = {
                        'latency_ms': latency_ms,
                        'packet_loss_rate': packet_loss,
                        'jitter_ms': jitter_ms
                    }
                
                return client.connection_quality
        except Exception as e:
            print(f"⚠️ Failed to measure connection quality: {e}")
        
        return {'latency_ms': 0, 'packet_loss_rate': 1.0, 'jitter_ms': 0}
    
    def get_client_info(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Get client information"""
        with self._lock:
            if client_id not in self.clients:
                return None
            
            client = self.clients[client_id]
            return {
                'client_id': client.client_id,
                'ip_address': client.ip_address,
                'hostname': client.hostname,
                'username': client.username,
                'status': client.status,
                'last_seen': datetime.fromtimestamp(client.last_seen).isoformat(),
                'round_number': client.round_number,
                'connection_quality': client.connection_quality,
                'metrics': client.metrics
            }
    
    def get_all_clients(self) -> List[Dict[str, Any]]:
        """Get all registered clients"""
        with self._lock:
            return [
                {
                    'client_id': c.client_id,
                    'ip_address': c.ip_address,
                    'hostname': c.hostname,
                    'status': c.status,
                    'last_seen': datetime.fromtimestamp(c.last_seen).isoformat(),
                    'connection_quality': c.connection_quality
                }
                for c in self.clients.values()
            ]
    
    def get_active_clients(self) -> List[int]:
        """Get list of active client IDs"""
        with self._lock:
            return [
                c.client_id for c in self.clients.values()
                if c.status in ('connected', 'training')
            ]
    
    def start_monitoring(self, interval: int = 30):
        """Start monitoring client connections"""
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        print("📡 Started physical client monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("📡 Stopped physical client monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Check each client's connection
                with self._lock:
                    client_ids = list(self.clients.keys())
                
                for client_id in client_ids:
                    # Measure connection quality
                    self.measure_connection_quality(client_id)
                    
                    # Check if client is still alive (timeout)
                    with self._lock:
                        if client_id in self.clients:
                            client = self.clients[client_id]
                            time_since_seen = time.time() - client.last_seen
                            
                            if time_since_seen > 120:  # 2 minutes timeout
                                client.status = "disconnected"
                                print(f"⚠️ Client {client_id} disconnected (timeout)")
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
            
            time.sleep(interval)


# Global instance
physical_client_manager = PhysicalClientManager()


def create_physical_client_manager(server_url: str = "http://localhost:5000"):
    """Factory function to create physical client manager"""
    global physical_client_manager
    physical_client_manager = PhysicalClientManager(server_url)
    return physical_client_manager


# Server-side metrics collection
class ServerMetricsCollector:
    """
    Collects metrics on the server side for communication with RPi clients
    """
    
    def __init__(self):
        self.round_metrics: List[Dict[str, Any]] = []
        self.client_latency_history: Dict[int, List[float]] = {}
    
    def record_client_update(
        self,
        client_id: int,
        round_number: int,
        metrics: Dict[str, Any]
    ):
        """Record metrics from a client update"""
        if client_id not in self.client_latency_history:
            self.client_latency_history[client_id] = []
        
        # Extract latency if available
        if 'network' in metrics and 'latency_ms' in metrics['network']:
            self.client_latency_history[client_id].append(
                metrics['network']['latency_ms']
            )
        
        # Keep only last 100 measurements
        if len(self.client_latency_history[client_id]) > 100:
            self.client_latency_history[client_id] = \
                self.client_latency_history[client_id][-100:]
    
    def get_client_latency_stats(self, client_id: int) -> Dict[str, float]:
        """Get latency statistics for a client"""
        if client_id not in self.client_latency_history:
            return {'avg': 0, 'min': 0, 'max': 0, 'std': 0}
        
        latencies = self.client_latency_history[client_id]
        if not latencies:
            return {'avg': 0, 'min': 0, 'max': 0, 'std': 0}
        
        import statistics
        return {
            'avg': statistics.mean(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get server network statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_received': net_io.bytes_received,
            'packets_sent': net_io.packets_sent,
            'packets_received': net_io.packets_received,
            'errors': net_io.errin + net_io.errout,
            'drops': net_io.dropin + net_io.dropout
        }


# Global server metrics collector
server_metrics_collector = ServerMetricsCollector()


if __name__ == "__main__":
    # Test the manager
    print("Testing Physical Client Manager...")
    
    manager = PhysicalClientManager()
    
    # Register a test client
    manager.register_client(
        client_id=0,
        ip_address="192.168.100.41",
        hostname="raspberry-pi-0",
        username="pi1"
    )
    
    # Get client info
    client_info = manager.get_client_info(0)
    print(f"Client info: {client_info}")
    
    # Measure connection quality
    quality = manager.measure_connection_quality(0)
    print(f"Connection quality: {quality}")
    
    # Get all clients
    all_clients = manager.get_all_clients()
    print(f"All clients: {all_clients}")