import os
import json
from datetime import datetime, timedelta
import jwt
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

DEFAULT_JWT_SECRET = 'multiorganai-jwt-super-secret-key-2026'

# ==========================================
# 1. USERS MODEL
# ==========================================
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(30), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    blood_group = db.Column(db.String(10), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), default='patient')  # patient, doctor, admin
    is_donor = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    predictions = db.relationship('PredictionRecord', backref='user', lazy=True, cascade='all, delete-orphan')
    reports = db.relationship('MedicalReport', backref='user', lazy=True, cascade='all, delete-orphan')
    uploaded_files = db.relationship('UploadedFile', backref='user', lazy=True, cascade='all, delete-orphan')
    notifications = db.relationship('Notification', backref='user', lazy=True, cascade='all, delete-orphan')
    donor_profile = db.relationship('Donor', backref='user', uselist=False, lazy=True)
    recipient_profile = db.relationship('Recipient', backref='user', uselist=False, lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_token(self, secret_key=None, expires_in=86400):
        """Generate JWT auth token valid for `expires_in` seconds (default 24h)."""
        secret = secret_key or os.environ.get('JWT_SECRET', DEFAULT_JWT_SECRET)
        payload = {
            'user_id': self.id,
            'email': self.email,
            'role': self.role,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, secret, algorithm='HS256')

    @staticmethod
    def verify_token(token, secret_key=None):
        """Verify and decode a JWT token, returns user_id or None."""
        secret = secret_key or os.environ.get('JWT_SECRET', DEFAULT_JWT_SECRET)
        try:
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            return payload.get('user_id')
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'full_name': self.full_name or self.username,
            'blood_group': self.blood_group,
            'age': self.age,
            'gender': self.gender,
            'role': self.role,
            'is_donor': self.is_donor,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }


# ==========================================
# 2. PREDICTIONS MODEL
# ==========================================
class PredictionRecord(db.Model):
    __tablename__ = 'prediction_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    guest_identifier = db.Column(db.String(100), nullable=True)
    organ_type = db.Column(db.String(30), nullable=False, index=True)  # 'kidney', 'liver', 'heart'
    input_parameters = db.Column(db.Text, nullable=False)  # JSON string
    risk_score = db.Column(db.Float, default=0.0)  # 0.0 - 100.0
    risk_level = db.Column(db.String(20), default='Low')  # 'Low', 'Moderate', 'High'
    prediction_label = db.Column(db.String(120), nullable=False)
    disease_detected = db.Column(db.String(10), default='No')  # 'Yes' or 'No'
    confidence = db.Column(db.Float, default=0.0)  # 0.0 - 100.0
    probability = db.Column(db.Float, default=0.0)  # 0.0 - 1.0 or 0.0 - 100.0
    model_name = db.Column(db.String(100), default='MultiOrganAI Ensemble')
    clinical_summary = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)  # JSON string
    pdf_filename = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Aliases and properties for requested API compatibility
    @property
    def organ(self):
        return self.organ_type.capitalize()

    @organ.setter
    def organ(self, val):
        self.organ_type = str(val).lower()

    @property
    def parameters(self):
        try:
            return json.loads(self.input_parameters) if self.input_parameters else {}
        except Exception:
            return {}

    @parameters.setter
    def parameters(self, val):
        self.input_parameters = json.dumps(val) if isinstance(val, (dict, list)) else str(val)

    @property
    def prediction(self):
        return self.prediction_label

    @prediction.setter
    def prediction(self, val):
        self.prediction_label = val

    @property
    def recommendation(self):
        try:
            recs = json.loads(self.recommendations) if self.recommendations else []
            return recs[0] if recs else self.clinical_summary
        except Exception:
            return self.clinical_summary

    def to_dict(self):
        try:
            inputs = json.loads(self.input_parameters) if self.input_parameters else {}
        except Exception:
            inputs = {}
        try:
            recs = json.loads(self.recommendations) if self.recommendations else []
        except Exception:
            recs = []

        return {
            'id': self.id,
            'user_id': self.user_id,
            'organ': self.organ_type.capitalize(),
            'organ_type': self.organ_type.capitalize(),
            'parameters': inputs,
            'input_parameters': inputs,
            'prediction': self.prediction_label,
            'prediction_label': self.prediction_label,
            'disease_detected': self.disease_detected,
            'confidence': round(self.confidence, 1),
            'probability': round(self.probability, 4) if self.probability else round(self.risk_score / 100.0, 4),
            'risk_score': round(self.risk_score, 1),
            'risk_level': self.risk_level,
            'model_name': self.model_name or 'MultiOrganAI Ensemble',
            'clinical_summary': self.clinical_summary,
            'recommendation': recs[0] if recs else self.clinical_summary,
            'recommendations': recs,
            'pdf_filename': self.pdf_filename,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'created_at_human': self.created_at.strftime('%b %d, %Y - %H:%M') if self.created_at else None
        }

# Alias for API terminology
Prediction = PredictionRecord


# ==========================================
# 3. MEDICAL REPORTS MODEL
# ==========================================
class MedicalReport(db.Model):
    __tablename__ = 'medical_reports'

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(80), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    guest_identifier = db.Column(db.String(100), nullable=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_records.id'), nullable=True, index=True)
    organ = db.Column(db.String(30), nullable=False)  # 'Kidney', 'Heart', 'Liver'
    parameters = db.Column(db.Text, nullable=False)  # JSON dictionary
    parameter_analysis = db.Column(db.Text, nullable=False)  # JSON list
    abnormal_parameters = db.Column(db.Text, nullable=True)  # JSON list
    normal_parameters = db.Column(db.Text, nullable=True)  # JSON list
    prediction = db.Column(db.String(120), nullable=False)
    disease_detected = db.Column(db.String(10), default='No')  # 'Yes' or 'No'
    confidence = db.Column(db.Float, default=0.0)  # e.g. 92.45
    risk_level = db.Column(db.String(30), nullable=False)  # 'Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'
    risk_score = db.Column(db.Float, default=0.0)  # 0.0 - 100.0
    model_used = db.Column(db.String(100), nullable=True)
    recommendations = db.Column(db.Text, nullable=True)  # JSON list
    generated_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    pdf_path = db.Column(db.String(255), nullable=True)

    prediction_record = db.relationship('PredictionRecord', backref=db.backref('medical_report', uselist=False))

    def to_dict(self):
        try:
            params = json.loads(self.parameters) if self.parameters else {}
        except Exception:
            params = {}
        try:
            param_analysis = json.loads(self.parameter_analysis) if self.parameter_analysis else []
        except Exception:
            param_analysis = []
        try:
            abnormal = json.loads(self.abnormal_parameters) if self.abnormal_parameters else []
        except Exception:
            abnormal = []
        try:
            normal = json.loads(self.normal_parameters) if self.normal_parameters else []
        except Exception:
            normal = []
        try:
            recs = json.loads(self.recommendations) if self.recommendations else []
        except Exception:
            recs = []

        return {
            'id': self.id,
            'report_id': self.report_id,
            'user_id': self.user_id,
            'guest_identifier': self.guest_identifier,
            'prediction_id': self.prediction_id,
            'organ': self.organ.capitalize(),
            'parameters': params,
            'parameter_analysis': param_analysis,
            'abnormal_parameters': abnormal,
            'normal_parameters': normal,
            'prediction': self.prediction,
            'disease_detected': self.disease_detected,
            'confidence': round(self.confidence, 2),
            'risk_level': self.risk_level,
            'risk_score': round(self.risk_score, 1),
            'model_used': self.model_used or 'MultiOrganAI Ensemble',
            'recommendations': recs,
            'generated_at': self.generated_at.strftime('%Y-%m-%d %H:%M:%S') if self.generated_at else None,
            'generated_at_human': self.generated_at.strftime('%b %d, %Y - %H:%M UTC') if self.generated_at else None,
            'generated_at_iso': self.generated_at.isoformat() if self.generated_at else None,
            'pdf_path': self.pdf_path
        }


# ==========================================
# 4. UPLOADED FILES MODEL
# ==========================================
class UploadedFile(db.Model):
    __tablename__ = 'uploaded_files'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # 'pdf', 'png', 'jpg', 'jpeg'
    file_size = db.Column(db.Integer, nullable=False, default=0)  # bytes
    extracted_parameters = db.Column(db.Text, nullable=True)  # JSON string
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        try:
            params = json.loads(self.extracted_parameters) if self.extracted_parameters else {}
        except Exception:
            params = {}

        return {
            'id': self.id,
            'user_id': self.user_id,
            'original_filename': self.original_filename,
            'stored_filename': self.stored_filename,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'extracted_parameters': params,
            'uploaded_at': self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S') if self.uploaded_at else None
        }


# ==========================================
# 5. DONORS MODEL
# ==========================================
class Donor(db.Model):
    __tablename__ = 'donors'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    donor_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    blood_group = db.Column(db.String(10), nullable=False, index=True)
    organ_offered = db.Column(db.String(50), nullable=False, index=True)  # 'Kidney', 'Liver', 'Heart'
    hospital_city = db.Column(db.String(100), nullable=False)
    contact_phone = db.Column(db.String(30), nullable=False)
    medical_clearance = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(30), default='Available', index=True)  # 'Available', 'Matched', 'Pledged'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Properties for API schema compatibility
    @property
    def full_name(self):
        return self.donor_name

    @full_name.setter
    def full_name(self, value):
        self.donor_name = value

    @property
    def organ(self):
        return self.organ_offered

    @organ.setter
    def organ(self, value):
        self.organ_offered = value

    @property
    def hospital(self):
        return self.hospital_city

    @hospital.setter
    def hospital(self, value):
        self.hospital_city = value

    @property
    def medical_status(self):
        return 'Cleared' if self.medical_clearance else 'Pending'

    @medical_status.setter
    def medical_status(self, value):
        val = str(value).lower()
        self.medical_clearance = True if val in ['cleared', 'true', 'yes', '1'] else False

    @property
    def availability(self):
        return self.status

    @availability.setter
    def availability(self, value):
        self.status = value

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'full_name': self.donor_name,
            'donor_name': self.donor_name,
            'age': self.age,
            'gender': self.gender,
            'blood_group': self.blood_group,
            'organ': self.organ_offered,
            'organ_offered': self.organ_offered,
            'hospital': self.hospital_city,
            'hospital_city': self.hospital_city,
            'contact_phone': self.contact_phone,
            'medical_status': 'Cleared' if self.medical_clearance else 'Pending',
            'medical_clearance': self.medical_clearance,
            'availability': self.status,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }


# ==========================================
# 6. ORGAN REQUESTS (RECIPIENTS) MODEL
# ==========================================
class Recipient(db.Model):
    __tablename__ = 'recipients'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    recipient_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    blood_group = db.Column(db.String(10), nullable=False, index=True)
    organ_needed = db.Column(db.String(50), nullable=False, index=True)  # 'Kidney', 'Liver', 'Heart'
    urgency_level = db.Column(db.String(20), nullable=False)  # 'Critical', 'High', 'Standard'
    hospital_name = db.Column(db.String(120), nullable=False)
    hospital_city = db.Column(db.String(100), nullable=False)
    contact_phone = db.Column(db.String(30), nullable=False)
    status = db.Column(db.String(30), default='Waiting', index=True)  # 'Waiting', 'Matched', 'Completed'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Properties for OrganRequest API compatibility
    @property
    def patient_name(self):
        return self.recipient_name

    @patient_name.setter
    def patient_name(self, value):
        self.recipient_name = value

    @property
    def required_organ(self):
        return self.organ_needed

    @required_organ.setter
    def required_organ(self, value):
        self.organ_needed = value

    @property
    def urgency(self):
        return self.urgency_level

    @urgency.setter
    def urgency(self, value):
        self.urgency_level = value

    @property
    def hospital(self):
        return self.hospital_name

    @hospital.setter
    def hospital(self, value):
        self.hospital_name = value

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'patient_name': self.recipient_name,
            'recipient_name': self.recipient_name,
            'age': self.age,
            'gender': self.gender,
            'blood_group': self.blood_group,
            'required_organ': self.organ_needed,
            'organ_needed': self.organ_needed,
            'urgency': self.urgency_level,
            'urgency_level': self.urgency_level,
            'hospital': self.hospital_name,
            'hospital_name': self.hospital_name,
            'hospital_city': self.hospital_city,
            'contact_phone': self.contact_phone,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }

# Full alias for OrganRequest
OrganRequest = Recipient


# ==========================================
# 7. ORGAN MATCHES MODEL
# ==========================================
class MatchRecord(db.Model):
    __tablename__ = 'match_records'

    id = db.Column(db.Integer, primary_key=True)
    donor_id = db.Column(db.Integer, db.ForeignKey('donors.id'), nullable=False, index=True)
    recipient_id = db.Column(db.Integer, db.ForeignKey('recipients.id'), nullable=False, index=True)
    organ_type = db.Column(db.String(50), nullable=False, index=True)
    compatibility_score = db.Column(db.Float, nullable=False)  # 0 - 100%
    blood_compatibility = db.Column(db.String(50), default='Compatible')
    urgency_score = db.Column(db.Float, default=0.0)
    match_notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(30), default='Under Review', index=True)  # 'Under Review', 'Approved', 'Completed'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    donor = db.relationship('Donor', backref='matches')
    recipient = db.relationship('Recipient', backref='matches')

    # Aliases
    @property
    def request_id(self):
        return self.recipient_id

    @request_id.setter
    def request_id(self, val):
        self.recipient_id = val

    @property
    def organ(self):
        return self.organ_type

    @organ.setter
    def organ(self, val):
        self.organ_type = val

    @property
    def match_score(self):
        return self.compatibility_score

    @match_score.setter
    def match_score(self, val):
        self.compatibility_score = val

    @property
    def match_status(self):
        return self.status

    @match_status.setter
    def match_status(self, val):
        self.status = val

    def to_dict(self):
        return {
            'id': self.id,
            'donor_id': self.donor_id,
            'request_id': self.recipient_id,
            'recipient_id': self.recipient_id,
            'organ': self.organ_type,
            'organ_type': self.organ_type,
            'blood_compatibility': self.blood_compatibility or 'Compatible',
            'match_score': round(self.compatibility_score, 1),
            'compatibility_score': round(self.compatibility_score, 1),
            'urgency_score': round(self.urgency_score, 1),
            'match_status': self.status,
            'status': self.status,
            'match_notes': self.match_notes,
            'donor': self.donor.to_dict() if self.donor else None,
            'recipient': self.recipient.to_dict() if self.recipient else None,
            'request': self.recipient.to_dict() if self.recipient else None,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

# Full alias for OrganMatch
OrganMatch = MatchRecord


# ==========================================
# 8. NOTIFICATIONS MODEL
# ==========================================
class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), default='info', index=True)  # 'prediction', 'report', 'donor', 'request', 'match', 'info'
    is_read = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'type': self.type,
            'is_read': self.is_read,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'created_at_human': self.created_at.strftime('%b %d, %Y - %H:%M') if self.created_at else None,
            'created_at_iso': self.created_at.isoformat() if self.created_at else None
        }


def notify(user_id, title, message, notif_type='info'):
    """Safe helper to create and commit a notification without breaking callers."""
    try:
        n = Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notif_type,
            is_read=False,
            created_at=datetime.utcnow()
        )
        db.session.add(n)
        db.session.commit()
        return n
    except Exception as e:
        db.session.rollback()
        print(f"[WARN] Failed to write notification: {e}")
        return None


# ==========================================
# INITIALIZATION & SEEDING
# ==========================================
def migrate_sqlite_schema():
    """Ensure SQLite schema has all newly introduced columns and tables."""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE users ADD COLUMN phone VARCHAR(30);",
        "ALTER TABLE users ADD COLUMN updated_at DATETIME;",
        "ALTER TABLE prediction_records ADD COLUMN disease_detected VARCHAR(10) DEFAULT 'No';",
        "ALTER TABLE prediction_records ADD COLUMN probability FLOAT DEFAULT 0.0;",
        "ALTER TABLE prediction_records ADD COLUMN model_name VARCHAR(100) DEFAULT 'MultiOrganAI Ensemble';",
        "ALTER TABLE donors ADD COLUMN updated_at DATETIME;",
        "ALTER TABLE recipients ADD COLUMN updated_at DATETIME;",
        "ALTER TABLE match_records ADD COLUMN blood_compatibility VARCHAR(50) DEFAULT 'Compatible';",
        "ALTER TABLE match_records ADD COLUMN urgency_score FLOAT DEFAULT 0.0;",
    ]
    for sql in migrations:
        try:
            db.session.execute(text(sql))
            db.session.commit()
        except Exception:
            db.session.rollback()

def init_db(app):
    db_dir = os.path.dirname(app.config['DATABASE_PATH'])
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_REPORTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()
        migrate_sqlite_schema()
        seed_demo_data()

def seed_demo_data():
    """Seed demo accounts, donors, recipients, predictions, and notifications if DB is fresh."""
    if User.query.first() is not None:
        return

    # Seed demo user
    demo_user = User(
        username='demo_user',
        email='demo@multiorganai.health',
        phone='+1-800-555-0199',
        full_name='Dr. Alex Morgan',
        blood_group='O+',
        age=36,
        gender='Male',
        role='doctor'
    )
    demo_user.set_password('demo1234')
    db.session.add(demo_user)
    db.session.commit()

    # Seed initial Donors
    donors = [
        Donor(donor_name='Sarah Jenkins', age=29, gender='Female', blood_group='O+', organ_offered='Kidney', hospital_city='New York', contact_phone='+1-555-0144', status='Available'),
        Donor(donor_name='David Kim', age=34, gender='Male', blood_group='A+', organ_offered='Liver', hospital_city='Boston', contact_phone='+1-555-0178', status='Available'),
        Donor(donor_name='Elena Rostova', age=42, gender='Female', blood_group='B+', organ_offered='Kidney', hospital_city='Chicago', contact_phone='+1-555-0199', status='Available'),
        Donor(donor_name='Marcus Bennett', age=27, gender='Male', blood_group='O-', organ_offered='Heart', hospital_city='Philadelphia', contact_phone='+1-555-0211', status='Available'),
        Donor(donor_name='Priya Sharma', age=31, gender='Female', blood_group='AB+', organ_offered='Liver', hospital_city='New York', contact_phone='+1-555-0322', status='Available'),
    ]
    for d in donors:
        db.session.add(d)

    # Seed initial Recipients
    recipients = [
        Recipient(recipient_name='Robert Hayes', age=54, gender='Male', blood_group='O+', organ_needed='Kidney', urgency_level='Critical', hospital_name='Metropolitan General Hospital', hospital_city='New York', contact_phone='+1-555-0891', status='Waiting'),
        Recipient(recipient_name='Maria Garcia', age=48, gender='Female', blood_group='A+', organ_needed='Liver', urgency_level='High', hospital_name='Boston Medical Center', hospital_city='Boston', contact_phone='+1-555-0844', status='Waiting'),
        Recipient(recipient_name='James Wilson', age=61, gender='Male', blood_group='O-', organ_needed='Heart', urgency_level='Critical', hospital_name='Penn Health Pavilion', hospital_city='Philadelphia', contact_phone='+1-555-0766', status='Waiting'),
        Recipient(recipient_name='Amina Patel', age=38, gender='Female', blood_group='B+', organ_needed='Kidney', urgency_level='Standard', hospital_name='Northwestern Memorial Hospital', hospital_city='Chicago', contact_phone='+1-555-0552', status='Waiting'),
    ]
    for r in recipients:
        db.session.add(r)

    # Sample demo prediction records
    sample_inputs_kidney = {
        'age': 52, 'bp': 90, 'sg': 1.015, 'al': 3, 'su': 1,
        'bgr': 160, 'bu': 58, 'sc': 2.4, 'sod': 132, 'pot': 4.9, 'hemo': 10.2, 'wbcc': 9800
    }
    rec1 = PredictionRecord(
        user_id=demo_user.id,
        organ_type='kidney',
        input_parameters=json.dumps(sample_inputs_kidney),
        risk_score=78.5,
        risk_level='High',
        prediction_label='Potential Chronic Kidney Condition Detected',
        disease_detected='Yes',
        confidence=89.2,
        probability=0.785,
        model_name='MultiOrganAI ExtraTrees Classifier',
        clinical_summary='Elevated serum creatinine (2.4 mg/dL) and marked proteinuria (Albumin +3) indicate glomerular impairment.',
        recommendations=json.dumps([
            'Immediate nephrology consultation recommended.',
            'Perform 24-hour urine protein quantification and renal ultrasound.',
            'Restrict dietary sodium (< 2,000 mg/day) and monitor potassium intake.',
            'Strict blood pressure control target < 130/80 mmHg.'
        ])
    )
    db.session.add(rec1)
    db.session.commit()

    # Seed initial notification
    notify(
        demo_user.id,
        'Welcome to MultiOrganAI Platform',
        'Your secure AI-Powered clinical diagnostic & organ matching account is ready.',
        'info'
    )
