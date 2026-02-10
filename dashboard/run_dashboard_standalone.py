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


def main():
    """Main entry point for standalone dashboard"""
    
    print("="*70)
    print("XFL-RPiLab Dashboard (Docker)")
    print("="*70)
    print("Starting dashboard on 0.0.0.0:5001")
    print("="*70 + "\n")
    
    run_dashboard(db_path="/app/logs/server_metrics.db", port=5001)

app = Flask(__name__)
CORS(app)
# Ajouter la route XFL set_mode
@app.route('/api/xfl/set_mode', methods=['POST'])
def set_xfl_mode():
    """Proxy to set XFL mode on FL server"""
    try:
        import requests
        data = request.get_json()
        
        response = requests.post(
            'http://server:5000/xfl/set_mode',
            json=data,
            timeout=5
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/xfl/statistics', methods=['GET'])
def get_xfl_statistics():
    """Get XFL statistics from server"""
    try:
        import requests
        response = requests.get('http://server:5000/xfl/statistics', timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    main()