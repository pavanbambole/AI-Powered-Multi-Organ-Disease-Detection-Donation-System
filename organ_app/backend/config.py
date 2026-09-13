import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'multiorganai-dev-secret-key-2026-secure')
    JWT_SECRET = os.environ.get('JWT_SECRET', 'multiorganai-jwt-super-secret-key-2026')
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
    DATABASE_PATH = os.path.join(PROJECT_ROOT, 'database', 'organ_app.db')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f"sqlite:///{DATABASE_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
    GENERATED_REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports', 'generated')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}

    # Medical confidence thresholds
    RISK_LEVEL_THRESHOLDS = {
        'low': 0.35,
        'moderate': 0.65,
        'high': 1.00
    }
