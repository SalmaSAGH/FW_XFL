"""
Client-side metrics collection
"""

import psutil
import time
import sys
from typing import Dict, Any
from collections import OrderedDict
import random


class MetricsCollector:
    """
    Collect system and training metrics for FL clients
    """
    
    def __init__(self, client_id: int, network_config: Dict[str, Any] = None):
        """
        Args:
            client_id: Unique identifier for this client
            network_config: Network configuration for latency simulation
        """
        self.client_id = client_id
        self.process = psutil.Process()
        self.start_time = None
        self.network_config = network_config or {}
        self.cpu_history = []  # For energy calculation
        
    def start_collection(self):
        """Start metrics collection"""
        self.start_time = time.time()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Collect current system metrics
        
        Returns:
            Dictionary with CPU, RAM metrics
        """
        # CPU metrics
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        memory_percent = self.process.memory_percent()
        
        # System-wide metrics
        system_cpu = psutil.cpu_percent(interval=0.1)
        system_memory = psutil.virtual_memory().percent
        
        return {
            "process_cpu_percent": round(cpu_percent, 2),
            "process_memory_mb": round(memory_mb, 2),
            "process_memory_percent": round(memory_percent, 2),
            "system_cpu_percent": round(system_cpu, 2),
            "system_memory_percent": round(system_memory, 2)
        }
    
    def calculate_model_size(self, model_weights: OrderedDict) -> Dict[str, float]:
        """
        Calculate size of model weights
        
        Args:
            model_weights: Model state dict
            
        Returns:
            Dictionary with model size metrics
        """
        import torch
        
        total_params = 0
        total_bytes = 0
        
        for name, param in model_weights.items():
            num_params = param.numel()
            num_bytes = param.element_size() * num_params
            
            total_params += num_params
            total_bytes += num_bytes
        
        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "total_mb": round(total_mb, 4),
            "num_layers": len(model_weights)
        }
    
    def get_training_metrics(
        self,
        loss: float,
        accuracy: float,
        training_time: float,
        num_samples: int,
        num_epochs: int
    ) -> Dict[str, Any]:
        """
        Organize training metrics
        
        Args:
            loss: Training loss
            accuracy: Training accuracy
            training_time: Time taken for training
            num_samples: Number of training samples
            num_epochs: Number of local epochs
            
        Returns:
            Dictionary with training metrics
        """
        samples_per_second = num_samples * num_epochs / training_time if training_time > 0 else 0
        
        return {
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 2),
            "training_time_sec": round(training_time, 2),
            "num_samples": num_samples,
            "num_epochs": num_epochs,
            "samples_per_second": round(samples_per_second, 2)
        }
    
    def get_network_metrics(
        self,
        bytes_sent: int,
        bytes_received: int,
        transmission_time: float
    ) -> Dict[str, Any]:
        """
        Calculate network metrics

        Args:
            bytes_sent: Number of bytes sent
            bytes_received: Number of bytes received
            transmission_time: Time taken for transmission

        Returns:
            Dictionary with network metrics
        """
        # Convert to MB
        mb_sent = bytes_sent / (1024 * 1024)
        mb_received = bytes_received / (1024 * 1024)

        # Calculate throughput (Mbps)
        if transmission_time > 0:
            throughput_mbps = (bytes_sent * 8) / (transmission_time * 1_000_000)
        else:
            throughput_mbps = 0

        # Simulate latency based on network config
        latency_ms = self.simulate_latency()
        packet_loss_rate = self.simulate_packet_loss()
        jitter_ms = self.simulate_jitter()

        return {
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "mb_sent": round(mb_sent, 4),
            "mb_received": round(mb_received, 4),
            "transmission_time_sec": round(transmission_time, 2),
            "throughput_mbps": round(throughput_mbps, 2),
            "latency_ms": round(latency_ms, 2),
            "packet_loss_rate": round(packet_loss_rate, 4),
            "jitter_ms": round(jitter_ms, 2)
        }

    def simulate_latency(self) -> float:
        """
        Simulate network latency based on configuration

        Returns:
            Latency in milliseconds
        """
        if not self.network_config.get('simulate_constraints', False):
            return 0.0

        base_latency = self.network_config.get('latency_ms', 0)
        std_dev = self.network_config.get('latency_std_ms', 0)

        # Add random variation
        latency = random.gauss(base_latency, std_dev)
        return max(0, latency)  # Ensure non-negative

    def simulate_packet_loss(self) -> float:
        """
        Simulate packet loss rate

        Returns:
            Packet loss rate (0.0 to 1.0)
        """
        if not self.network_config.get('simulate_constraints', False):
            return 0.0

        loss_rate = self.network_config.get('packet_loss_rate', 0.0)
        # Add some randomness
        variation = random.uniform(-0.01, 0.01)
        return max(0.0, min(1.0, loss_rate + variation))

    def simulate_jitter(self) -> float:
        """
        Simulate network jitter

        Returns:
            Jitter in milliseconds
        """
        if not self.network_config.get('simulate_constraints', False):
            return 0.0

        base_jitter = self.network_config.get('jitter_ms', 0.0)
        # Add random variation
        jitter = random.gauss(base_jitter, base_jitter * 0.1)
        return max(0, jitter)

    def estimate_energy_consumption(self, duration_sec: float = None) -> Dict[str, float]:
        """
        Estimate energy consumption based on CPU usage history

        Args:
            duration_sec: Duration to estimate for (uses elapsed time if None)

        Returns:
            Dictionary with energy metrics
        """
        if duration_sec is None:
            duration_sec = time.time() - self.start_time if self.start_time else 0

        # Get current CPU usage
        current_cpu = self.process.cpu_percent(interval=0.1)
        self.cpu_history.append((time.time(), current_cpu))

        # Keep only recent history (last 60 seconds)
        cutoff_time = time.time() - 60
        self.cpu_history = [(t, cpu) for t, cpu in self.cpu_history if t > cutoff_time]

        # Calculate average CPU usage
        if self.cpu_history:
            avg_cpu = sum(cpu for _, cpu in self.cpu_history) / len(self.cpu_history)
        else:
            avg_cpu = current_cpu

        # Estimate power consumption (simplified model)
        # Base power + CPU-dependent power
        base_power_watts = 10.0  # Base system power
        cpu_power_watts = (avg_cpu / 100.0) * 50.0  # Max CPU power draw
        total_power_watts = base_power_watts + cpu_power_watts

        # Calculate energy consumption (Wh)
        energy_wh = (total_power_watts * duration_sec) / 3600.0

        # Convert to Joules
        energy_joules = energy_wh * 3600.0

        return {
            "energy_joules": round(energy_joules, 2),
            "energy_wh": round(energy_wh, 4),
            "avg_power_watts": round(total_power_watts, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "duration_sec": round(duration_sec, 2)
        }
    
    def collect_full_metrics(
        self,
        training_metrics: Dict[str, Any],
        model_weights: OrderedDict,
        network_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Collect all metrics for this round

        Args:
            training_metrics: Training metrics from LocalTrainer
            model_weights: Model weights
            network_metrics: Network metrics (optional)

        Returns:
            Complete metrics dictionary
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        metrics = {
            "client_id": self.client_id,
            "timestamp": time.time(),
            "elapsed_time_sec": round(elapsed_time, 2),

            # System metrics
            "system": self.get_system_metrics(),

            # Model metrics
            "model": self.calculate_model_size(model_weights),

            # Training metrics
            "training": training_metrics
        }

        # Add network metrics if provided
        if network_metrics:
            metrics["network"] = network_metrics

        # Add energy metrics
        metrics["energy"] = self.estimate_energy_consumption(elapsed_time)

        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """
        Pretty print metrics
        
        Args:
            metrics: Metrics dictionary
        """
        print(f"\n📊 Metrics for Client {self.client_id}:")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics['timestamp']))}")
        print(f"   Elapsed Time: {metrics['elapsed_time_sec']}s")
        
        print(f"\n   🖥️  System:")
        for key, value in metrics['system'].items():
            print(f"      {key}: {value}")
        
        print(f"\n   🧠 Model:")
        for key, value in metrics['model'].items():
            print(f"      {key}: {value}")
        
        print(f"\n   📈 Training:")
        for key, value in metrics['training'].items():
            print(f"      {key}: {value}")
        
        if 'network' in metrics:
            print(f"\n   🌐 Network:")
            for key, value in metrics['network'].items():
                print(f"      {key}: {value}")

        if 'energy' in metrics:
            print(f"\n   ⚡ Energy:")
            for key, value in metrics['energy'].items():
                print(f"      {key}: {value}")


# Test function
if __name__ == "__main__":
    """Test metrics collection"""
    print("🧪 Testing MetricsCollector...\n")
    
    import torch
    from model import create_model
    
    # Create metrics collector
    collector = MetricsCollector(client_id=0)
    collector.start_collection()
    
    # Simulate some work
    print("📊 Simulating training...")
    time.sleep(1)
    
    # Get system metrics
    print("\n📊 Collecting system metrics...")
    system_metrics = collector.get_system_metrics()
    print("System metrics:")
    for key, value in system_metrics.items():
        print(f"   {key}: {value}")
    
    # Create a model and calculate size
    print("\n📊 Calculating model size...")
    model = create_model("SimpleCNN", num_classes=10)
    model_weights = model.state_dict()
    model_size = collector.calculate_model_size(model_weights)
    print("Model size metrics:")
    for key, value in model_size.items():
        print(f"   {key}: {value}")
    
    # Simulate training metrics
    print("\n📊 Creating training metrics...")
    training_metrics = collector.get_training_metrics(
        loss=0.5,
        accuracy=85.5,
        training_time=10.5,
        num_samples=12000,
        num_epochs=1
    )
    
    # Simulate network metrics
    print("\n📊 Creating network metrics...")
    network_metrics = collector.get_network_metrics(
        bytes_sent=1_500_000,
        bytes_received=1_500_000,
        transmission_time=2.5
    )
    
    # Collect full metrics
    print("\n📊 Collecting full metrics...")
    full_metrics = collector.collect_full_metrics(
        training_metrics=training_metrics,
        model_weights=model_weights,
        network_metrics=network_metrics
    )
    
    # Print metrics
    collector.print_metrics(full_metrics)
    
    print("\n✅ All MetricsCollector tests passed!")