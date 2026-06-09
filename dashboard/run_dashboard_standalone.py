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
from db_config import DB_URL


def main():
    """Main entry point for standalone dashboard"""

    print("="*70)
    print("XFL-RPiLab Dashboard (Docker)")
    print("="*70)
    print("Starting dashboard on 0.0.0.0:5001")
    print("="*70 + "\n")

    # Use DB_URL from centralized config
    db_url = DB_URL
    print(f"Using DATABASE_URL: {db_url}")
    run_dashboard(db_url=db_url, port=5001)

if __name__ == "__main__":
    main()
