from functools import wraps
from flask import session, redirect, url_for, flash, request, jsonify

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_user():
    """Retrieve current user from Bearer JWT token or Flask session."""
    from database import User
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
        user_id = User.verify_token(token)
        if user_id:
            return User.query.get(user_id)
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def login_required(f):
    """Session or token-based authentication check."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            if request.path.startswith('/api/') or request.is_json:
                return jsonify({'success': False, 'error': 'Authentication required. Please login.'}), 401
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def auth_required(f):
    """Decorator that provides current_user if expected in view args."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'Authentication required. Please provide a Bearer token or login.'}), 401
        if 'current_user' in f.__code__.co_varnames:
            return f(current_user=user, *args, **kwargs)
        return f(*args, **kwargs)
    return decorated_function

def get_badge_class(risk_level):
    level = str(risk_level).lower()
    if 'high' in level or 'critical' in level:
        return 'badge-critical'
    if 'mod' in level or 'warn' in level:
        return 'badge-warning'
    return 'badge-normal'

def get_organ_color(organ):
    organ = str(organ).lower()
    if organ == 'kidney':
        return '#0284c7'  # Blue/cyan
    elif organ == 'heart':
        return '#ef4444'  # Red/rose
    elif organ == 'liver':
        return '#d97706'  # Amber/brown
    return '#10b981'
