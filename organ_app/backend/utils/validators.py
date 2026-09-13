import re

VALID_BLOOD_GROUPS = {'O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'}
VALID_ORGANS = {'kidney', 'liver'}

NUMERICAL_BOUNDS = {
    # Kidney
    'age': (1, 120),
    'bp': (40, 240),
    'sg': (1.000, 1.040),
    'al': (0, 5),
    'su': (0, 5),
    'bgr': (30, 600),
    'bu': (5, 350),
    'sc': (0.1, 25.0),
    'sod': (90, 180),
    'pot': (1.5, 9.5),
    'hemo': (2.0, 24.0),
    'pcv': (10.0, 70.0),
    'wbcc': (1000, 45000),
    'rbcc': (1.0, 10.0),

    # Heart
    'sex': (0, 1),
    'trestbps': (60, 240),
    'chol': (80, 600),
    'thalach': (50, 240),
    'oldpeak': (0.0, 10.0),
    'cp': (0, 3),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'exang': (0, 1),
    'slope': (0, 2),
    'ca': (0, 4),
    'thal': (0, 3),
    'hdl': (10, 150),
    'ldl': (10, 350),
    'triglycerides': (20, 1000),
    'heart_rate': (30, 220),

    # Liver
    'gender': (0, 1),
    'total_bilirubin': (0.1, 40.0),
    'direct_bilirubin': (0.0, 25.0),
    'alkaline_phosphotase': (20, 2500),
    'alamine_aminotransferase': (5, 1200),
    'aspartate_aminotransferase': (5, 1200),
    'total_proteins': (1.0, 14.0),
    'albumin': (0.5, 8.0),
    'albumin_and_globulin_ratio': (0.1, 4.0)
}

def validate_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email.strip()))

def validate_blood_group(bg):
    if not bg:
        return False
    return bg.strip().upper() in VALID_BLOOD_GROUPS

def validate_organ_type(organ):
    if not organ:
        return False
    return organ.strip().lower() in VALID_ORGANS

def validate_medical_inputs(organ_type, form_data):
    """
    Validates submitted medical inputs against physiological limits.
    Returns: (is_valid, errors_dict, cleaned_data)
    """
    errors = {}
    cleaned = {}

    for key, val in form_data.items():
        if key not in NUMERICAL_BOUNDS:
            cleaned[key] = val
            continue

        if val is None or str(val).strip() == '':
            continue

        try:
            num_val = float(val)
            min_v, max_v = NUMERICAL_BOUNDS[key]
            if num_val < min_v or num_val > max_v:
                errors[key] = f"Value {num_val} is outside clinical boundary [{min_v}, {max_v}]."
            else:
                cleaned[key] = num_val
        except (ValueError, TypeError):
            errors[key] = f"Invalid numeric parameter: {val}"

    return len(errors) == 0, errors, cleaned
