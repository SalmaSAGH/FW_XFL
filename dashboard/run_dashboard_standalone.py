"""
Standalone dashboard script for Docker deployment
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.dashboard import run_dashboard


def main():
    """Main entry point for standalone dashboard"""
    
    print("="*70)
    print("XFL-RPiLab Dashboard (Docker)")
    print("="*70)
    print("Starting dashboard on 0.0.0.0:5001")
    print("="*70 + "\n")
    
    run_dashboard(db_path="/app/logs/server_metrics.db", port=5001)


if __name__ == "__main__":
    main()