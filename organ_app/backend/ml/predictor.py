import os
import sys
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import automl_pipeline
    ClinicalPipelineWrapper = automl_pipeline.ClinicalPipelineWrapper
except ImportError:
    from ml import automl_pipeline
    ClinicalPipelineWrapper = automl_pipeline.ClinicalPipelineWrapper

sys.modules['automl_pipeline'] = automl_pipeline
sys.modules['train_template'] = automl_pipeline
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except (ImportError, AttributeError):
    pass

MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))

_MODEL_CACHE = {}

ORGAN_CONFIGS = {
    'kidney': {
        'model_file': 'kidney_model.pkl',
        'features': ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wbcc'],
        'defaults': {
            'age': 45.0, 'bp': 80.0, 'sg': 1.020, 'al': 0.0, 'su': 0.0,
            'bgr': 110.0, 'bu': 32.0, 'sc': 0.9, 'sod': 138.0, 'pot': 4.3, 'hemo': 14.5, 'wbcc': 7500.0
        }
    },
    'liver': {
        'model_file': 'liver_model.pkl',
        'features': [
            'age', 'gender', 'total_bilirubin', 'direct_bilirubin',
            'alkaline_phosphotase', 'alamine_aminotransferase',
            'aspartate_aminotransferase', 'total_proteins', 'albumin',
            'albumin_and_globulin_ratio'
        ],
        'defaults': {
            'age': 42.0, 'gender': 1.0, 'total_bilirubin': 0.9, 'direct_bilirubin': 0.2,
            'alkaline_phosphotase': 200.0, 'alamine_aminotransferase': 28.0,
            'aspartate_aminotransferase': 30.0, 'total_proteins': 7.0, 'albumin': 4.0,
            'albumin_and_globulin_ratio': 1.2
        }
    }
}

def get_model(organ_type):
    organ_key = organ_type.lower()
    if organ_key not in ORGAN_CONFIGS:
        raise ValueError(f"Unknown organ type: {organ_type}. Expected one of {list(ORGAN_CONFIGS.keys())}")

    if organ_key in _MODEL_CACHE:
        return _MODEL_CACHE[organ_key]

    model_file = ORGAN_CONFIGS[organ_key]['model_file']
    model_path = os.path.join(MODELS_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    payload = joblib.load(model_path)
    _MODEL_CACHE[organ_key] = payload
    return payload

def predict_organ(organ_type, input_dict):
    organ_key = organ_type.lower()
    config = ORGAN_CONFIGS.get(organ_key)
    if not config:
        raise ValueError(f"Invalid organ: {organ_type}")

    payload = get_model(organ_key)
    pipeline = payload['pipeline']
    features = config['features']
    defaults = config['defaults']

    # Normalize inputs
    cleaned_input = {}
    for feat in features:
        val = input_dict.get(feat)
        if val is None or val == '':
            cleaned_input[feat] = defaults[feat]
        else:
            try:
                cleaned_input[feat] = float(val)
            except (ValueError, TypeError):
                cleaned_input[feat] = defaults[feat]

    df_input = pd.DataFrame([cleaned_input], columns=features)

    # Predict probabilities
    probs = pipeline.predict_proba(df_input)[0]
    prob_disease = float(probs[1]) if len(probs) > 1 else float(probs[0])
    threshold = getattr(pipeline, 'threshold', 0.5)
    is_positive = bool(prob_disease >= threshold)
    risk_score = round(prob_disease * 100, 1)

    # Model metadata
    model_name = payload.get('model_name', 'AutoML Champion')
    if hasattr(pipeline, 'model_name') and pipeline.model_name:
        model_name = pipeline.model_name

    # Risk tiers according to user specification:
    # 0–34% -> Low Risk, 35–64% -> Medium Risk, 65–84% -> High Risk, 85–100% -> Critical Risk
    if risk_score >= 85.0:
        risk_level = 'Critical Risk'
        label = f'Critical Risk of {organ_key.capitalize()} Disease Detected'
        confidence = round(prob_disease * 100, 2)
    elif risk_score >= 65.0:
        risk_level = 'High Risk'
        label = f'{organ_key.capitalize()} Disease Risk Detected'
        confidence = round(prob_disease * 100, 2)
    elif risk_score >= 35.0:
        risk_level = 'Medium Risk'
        label = f'Moderate / Borderline Risk of {organ_key.capitalize()} Condition'
        confidence = round(prob_disease * 100, 2)
    else:
        risk_level = 'Low Risk'
        label = f'Normal / Healthy {organ_key.capitalize()} Function'
        confidence = round((1.0 - prob_disease) * 100, 2)

    disease_detected_str = 'Yes' if is_positive else 'No'

    return {
        'organ': organ_key,
        'organ_title': organ_key.capitalize(),
        'is_disease_detected': is_positive,
        'disease_detected': disease_detected_str,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'confidence': confidence,
        'label': label,
        'model_name': model_name,
        'cleaned_inputs': cleaned_input
    }
