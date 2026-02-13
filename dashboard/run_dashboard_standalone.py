"""
Standalone dashboard script for Docker deployment
"""

import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.dashboard import run_dashboard
from config.config_parser import load_config


def main():
    """Main entry point for standalone dashboard"""

    print("="*70)
    print("XFL-RPiLab Dashboard (Docker)")
    print("="*70)
    print("Starting dashboard on 0.0.0.0:5001")
    print("="*70 + "\n")

    run_dashboard(db_url="postgresql://postgres:newpassword@postgres:5432/xfl_metrics", port=5001)

if __name__ == "__main__":
    main()
