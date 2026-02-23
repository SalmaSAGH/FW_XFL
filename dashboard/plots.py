"""
Plotting and visualization module for XFL-RPiLab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Any


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """
    Visualize FL experiment results from database
    """

    def __init__(self, db_url: str = "postgresql://postgres:newpassword@localhost:5432/xfl_metrics", output_dir: str = "results"):
        """
        Args:
            db_url: PostgreSQL database URL
            output_dir: Directory to save plots
        """
        self.db_url = db_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ ResultsVisualizer initialized")
        print(f"   Database: {db_url}")
        print(f"   Output directory: {output_dir}")
    
    def load_round_metrics(self) -> pd.DataFrame:
        """Load round-level metrics into DataFrame"""
        conn = psycopg2.connect(self.db_url)
        df = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_number", conn)
        conn.close()
        return df

    def load_client_metrics(self) -> pd.DataFrame:
        """Load client-level metrics into DataFrame"""
        conn = psycopg2.connect(self.db_url)
        df = pd.read_sql_query("SELECT * FROM client_metrics ORDER BY round_number, client_id", conn)
        conn.close()
        return df
    
    def plot_accuracy_evolution(self, save: bool = True, show: bool = False):
        """
        Plot global model accuracy evolution across rounds
        
        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_round_metrics()
        
        if df.empty or 'global_test_accuracy' not in df.columns:
            print("⚠️  No accuracy data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot accuracy
        ax.plot(df['round_number'], df['global_test_accuracy'], 
                marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Global Test Accuracy')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Global Model Accuracy Evolution', fontsize=14, fontweight='bold', pad=20)
        
        # Set y-axis limits
        if not df['global_test_accuracy'].isna().all():
            min_acc = df['global_test_accuracy'].min()
            max_acc = df['global_test_accuracy'].max()
            ax.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))
        
        # Add legend
        ax.legend(loc='lower right', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "accuracy_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_evolution(self, save: bool = True, show: bool = False):
        """
        Plot global model loss evolution across rounds
        
        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_round_metrics()
        
        if df.empty or 'global_test_loss' not in df.columns:
            print("⚠️  No loss data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot loss
        ax.plot(df['round_number'], df['global_test_loss'], 
                marker='s', linewidth=2, markersize=8, color='#A23B72', label='Global Test Loss')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Global Model Loss Evolution', fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "loss_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_client_training_time(self, save: bool = True, show: bool = False):
        """
        Plot client training time across rounds
        
        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()
        
        if df.empty or 'training_time_sec' not in df.columns:
            print("⚠️  No training time data available")
            return
        
        # Calculate average training time per round
        avg_time = df.groupby('round_number')['training_time_sec'].mean().reset_index()
        std_time = df.groupby('round_number')['training_time_sec'].std().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average with error bars
        ax.errorbar(avg_time['round_number'], avg_time['training_time_sec'],
                   yerr=std_time['training_time_sec'], marker='o', linewidth=2,
                   markersize=8, capsize=5, color='#F18F01', label='Avg Training Time')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Client Training Time per Round', fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "training_time.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_system_metrics(self, save: bool = True, show: bool = False):
        """
        Plot system metrics (CPU and memory usage)
        
        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()
        
        if df.empty:
            print("⚠️  No client metrics available")
            return
        
        # Calculate averages per round
        avg_metrics = df.groupby('round_number').agg({
            'cpu_percent': 'mean',
            'memory_mb': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # CPU Usage
        if 'cpu_percent' in avg_metrics.columns and not avg_metrics['cpu_percent'].isna().all():
            ax1.plot(avg_metrics['round_number'], avg_metrics['cpu_percent'],
                    marker='o', linewidth=2, markersize=8, color='#06A77D')
            ax1.set_xlabel('FL Round', fontsize=11, fontweight='bold')
            ax1.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
            ax1.set_title('Average CPU Usage', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Memory Usage
        if 'memory_mb' in avg_metrics.columns and not avg_metrics['memory_mb'].isna().all():
            ax2.plot(avg_metrics['round_number'], avg_metrics['memory_mb'],
                    marker='s', linewidth=2, markersize=8, color='#D00000')
            ax2.set_xlabel('FL Round', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
            ax2.set_title('Average Memory Usage', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "system_metrics.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_client_performance_comparison(self, save: bool = True, show: bool = False):
        """
        Compare performance across different clients

        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()

        if df.empty or 'training_accuracy' not in df.columns:
            print("⚠️  No client accuracy data available")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Get unique clients
        clients = df['client_id'].unique()

        # Plot each client's accuracy
        for client_id in clients:
            client_data = df[df['client_id'] == client_id]
            ax.plot(client_data['round_number'], client_data['training_accuracy'],
                   marker='o', linewidth=2, markersize=6, label=f'Client {client_id}', alpha=0.7)

        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Client Training Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "client_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_latency_evolution(self, save: bool = True, show: bool = False):
        """
        Plot network latency evolution across rounds

        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()

        if df.empty or 'latency_ms' not in df.columns:
            print("⚠️  No latency data available")
            return

        # Calculate average latency per round
        avg_latency = df.groupby('round_number')['latency_ms'].mean().reset_index()
        std_latency = df.groupby('round_number')['latency_ms'].std().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot average with error bars
        ax.errorbar(avg_latency['round_number'], avg_latency['latency_ms'],
                   yerr=std_latency['latency_ms'], marker='o', linewidth=2,
                   markersize=8, capsize=5, color='#FF6B6B', label='Avg Latency')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Labels and title
        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Network Latency Evolution', fontsize=14, fontweight='bold', pad=20)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        # Tight layout
        plt.tight_layout()

        if save:
            output_path = self.output_dir / "latency_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_energy_consumption(self, save: bool = True, show: bool = False):
        """
        Plot energy consumption evolution across rounds

        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()

        if df.empty or 'energy_wh' not in df.columns:
            print("⚠️  No energy data available")
            return

        # Calculate average energy consumption per round
        avg_energy = df.groupby('round_number')['energy_wh'].mean().reset_index()
        std_energy = df.groupby('round_number')['energy_wh'].std().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot average with error bars
        ax.errorbar(avg_energy['round_number'], avg_energy['energy_wh'],
                   yerr=std_energy['energy_wh'], marker='s', linewidth=2,
                   markersize=8, capsize=5, color='#4ECDC4', label='Avg Energy Consumption')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Labels and title
        ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
        ax.set_title('Client Energy Consumption Evolution', fontsize=14, fontweight='bold', pad=20)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        # Tight layout
        plt.tight_layout()

        if save:
            output_path = self.output_dir / "energy_consumption.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_network_metrics(self, save: bool = True, show: bool = False):
        """
        Plot network metrics (packet loss and jitter)

        Args:
            save: Save plot to file
            show: Display plot
        """
        df = self.load_client_metrics()

        if df.empty:
            print("⚠️  No client metrics available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Packet Loss Rate
        if 'packet_loss_rate' in df.columns and not df['packet_loss_rate'].isna().all():
            avg_loss = df.groupby('round_number')['packet_loss_rate'].mean().reset_index()
            ax1.plot(avg_loss['round_number'], avg_loss['packet_loss_rate'] * 100,
                    marker='o', linewidth=2, markersize=8, color='#45B7D1')
            ax1.set_xlabel('FL Round', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Packet Loss Rate (%)', fontsize=11, fontweight='bold')
            ax1.set_title('Average Packet Loss Rate', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # Jitter
        if 'jitter_ms' in df.columns and not df['jitter_ms'].isna().all():
            avg_jitter = df.groupby('round_number')['jitter_ms'].mean().reset_index()
            ax2.plot(avg_jitter['round_number'], avg_jitter['jitter_ms'],
                    marker='s', linewidth=2, markersize=8, color='#FFA07A')
            ax2.set_xlabel('FL Round', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Jitter (ms)', fontsize=11, fontweight='bold')
            ax2.set_title('Average Network Jitter', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "network_metrics.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_all_plots(self, show: bool = False):
        """
        Generate all available plots
        
        Args:
            show: Display plots interactively
        """
        print("\n" + "="*70)
        print("Generating All Plots")
        print("="*70 + "\n")
        
        try:
            self.plot_accuracy_evolution(save=True, show=show)
        except Exception as e:
            print(f"⚠️  Could not generate accuracy plot: {e}")
        
        try:
            self.plot_loss_evolution(save=True, show=show)
        except Exception as e:
            print(f"⚠️  Could not generate loss plot: {e}")
        
        try:
            self.plot_client_training_time(save=True, show=show)
        except Exception as e:
            print(f"⚠️  Could not generate training time plot: {e}")
        
        try:
            self.plot_system_metrics(save=True, show=show)
        except Exception as e:
            print(f"⚠️  Could not generate system metrics plot: {e}")
        
        try:
            self.plot_client_performance_comparison(save=True, show=show)
        except Exception as e:
            print(f"⚠️  Could not generate client comparison plot: {e}")
        
        print(f"\n✅ All plots generated successfully!")
        print(f"📁 Plots saved to: {self.output_dir}")


# Test function
if __name__ == "__main__":
    """Generate plots from experiment results"""
    print("🧪 Generating plots from experiment results...\n")

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

    try:
        visualizer = ResultsVisualizer(
            db_url=db_url,
            output_dir="results"
        )

        visualizer.generate_all_plots(show=False)

        print("\n✅ Visualization complete!")

    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        print("   Make sure PostgreSQL is running and the database exists.")
        print("   For Docker: docker-compose up -d postgres")
        import traceback
        traceback.print_exc()
