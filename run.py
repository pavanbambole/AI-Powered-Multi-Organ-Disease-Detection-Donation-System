#!/usr/bin/env python3
"""
MultiOrganAI Platform Launcher
Run with: python run.py
"""
import sys
import os

# Add backend to python path
project_root = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(project_root, 'organ_app', 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================")
    print(f" MultiOrganAI System Starting on http://127.0.0.1:{port}")
    print(f" Demo Dashboard: http://127.0.0.1:{port}/demo")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
