import re

PATTERNS = {
    # Kidney
    'sc': [r'(?:serum\s+)?creatinine\s*[:=-]?\s*([0-9.]+)', r's\.?\s*creat\w*\s*[:=-]?\s*([0-9.]+)'],
    'bu': [r'blood\s+urea\s*[:=-]?\s*([0-9.]+)', r'\burea\s*[:=-]?\s*([0-9.]+)'],
    'bgr': [r'(?:random\s+)?blood\s+glucose\s*[:=-]?\s*([0-9.]+)', r'\bbgr\s*[:=-]?\s*([0-9.]+)', r'random\s+sugar\s*[:=-]?\s*([0-9.]+)'],
    'bp': [r'blood\s+pressure\s*[:=-]?\s*(?:[0-9]+)\s*/\s*([0-9]+)', r'\bbp\s*[:=-]?\s*(?:[0-9]+)\s*/\s*([0-9]+)'],
    'hemo': [r'h(?:a)?emoglobin\s*[:=-]?\s*([0-9.]+)', r'\bhb\s*[:=-]?\s*([0-9.]+)'],
    'al': [r'(?:urine\s+)?albumin\s*[:=-]?\s*\+?([0-9]+)', r'\balb\s*[:=-]?\s*\+?([0-9]+)'],
    'su': [r'(?:urine\s+)?sugar\s*[:=-]?\s*\+?([0-9]+)'],
    'sod': [r'(?:serum\s+)?sodium\s*[:=-]?\s*([0-9.]+)', r'\bna\+?\s*[:=-]?\s*([0-9.]+)'],
    'pot': [r'(?:serum\s+)?potassium\s*[:=-]?\s*([0-9.]+)', r'\bk\+?\s*[:=-]?\s*([0-9.]+)'],
    'sg': [r'specific\s+gravity\s*[:=-]?\s*([1-9]\.[0-9]{2,3})', r'\bsg\s*[:=-]?\s*([1-9]\.[0-9]{2,3})'],
    'pcv': [r'packed\s+cell\s+volume\s*[:=-]?\s*([0-9.]+)', r'\bpcv\s*[:=-]?\s*([0-9.]+)'],
    'wbcc': [r'(?:total\s+)?wbc\s*(?:count)?\s*[:=-]?\s*([0-9]+)', r'leukocytes?\s*[:=-]?\s*([0-9]+)'],
    'rbcc': [r'(?:total\s+)?rbc\s*(?:count)?\s*[:=-]?\s*([0-9.]+)', r'erythrocytes?\s*[:=-]?\s*([0-9.]+)'],

    # Heart
    'trestbps': [r'resting\s+blood\s+pressure\s*[:=-]?\s*([0-9]+)', r'resting\s+bp\s*[:=-]?\s*([0-9]+)', r'sys(?:tolic)?\s*bp\s*[:=-]?\s*([0-9]+)'],
    'chol': [r'(?:total\s+)?cholesterol\s*[:=-]?\s*([0-9.]+)', r'\bchol\s*[:=-]?\s*([0-9.]+)'],
    'thalach': [r'max(?:imum)?\s*heart\s*rate\s*[:=-]?\s*([0-9]+)', r'\bmax\s*hr\s*[:=-]?\s*([0-9]+)', r'\bthalach\s*[:=-]?\s*([0-9]+)'],
    'oldpeak': [r'st\s+depression\s*[:=-]?\s*([0-9.]+)', r'oldpeak\s*[:=-]?\s*([0-9.]+)'],
    'fbs': [r'fasting\s+blood\s+sugar\s*[:=-]?\s*([0-9.]+)', r'fasting\s+glucose\s*[:=-]?\s*([0-9.]+)'],
    'cp': [r'chest\s+pain(?:\s+type)?\s*[:=-]?\s*([0-3])'],
    'hdl': [r'\bhdl(?:\s+cholesterol)?\s*[:=-]?\s*([0-9.]+)', r'high-density\s+lipoprotein\s*[:=-]?\s*([0-9.]+)'],
    'ldl': [r'\bldl(?:\s+cholesterol)?\s*[:=-]?\s*([0-9.]+)', r'low-density\s+lipoprotein\s*[:=-]?\s*([0-9.]+)'],
    'triglycerides': [r'triglycerides?\s*[:=-]?\s*([0-9.]+)', r'\btg\s*[:=-]?\s*([0-9.]+)'],
    'heart_rate': [r'heart\s+rate\s*[:=-]?\s*([0-9]+)', r'\bpulse\s*[:=-]?\s*([0-9]+)'],

    # Liver
    'total_bilirubin': [r'total\s+bilirubin\s*[:=-]?\s*([0-9.]+)', r'\bt\.?\s*bili\w*\s*[:=-]?\s*([0-9.]+)'],
    'direct_bilirubin': [r'direct\s+bilirubin\s*[:=-]?\s*([0-9.]+)', r'\bd\.?\s*bili\w*\s*[:=-]?\s*([0-9.]+)'],
    'alkaline_phosphotase': [r'alkaline\s+phosphatase\s*[:=-]?\s*([0-9.]+)', r'\balp\s*[:=-]?\s*([0-9.]+)'],
    'alamine_aminotransferase': [r'(?:alt|sgpt)\s*[:=-]?\s*([0-9.]+)', r'alanine\s+aminotransferase\s*[:=-]?\s*([0-9.]+)'],
    'aspartate_aminotransferase': [r'(?:ast|sgot)\s*[:=-]?\s*([0-9.]+)', r'aspartate\s+aminotransferase\s*[:=-]?\s*([0-9.]+)'],
    'total_proteins': [r'total\s+protein(?:s)?\s*[:=-]?\s*([0-9.]+)', r'\bt\.?\s*protein\w*\s*[:=-]?\s*([0-9.]+)'],
    'albumin': [r'(?:serum\s+)?albumin\s*[:=-]?\s*([0-9.]+)', r'\bs\.?\s*alb\w*\s*[:=-]?\s*([0-9.]+)'],
    'albumin_and_globulin_ratio': [r'a/g\s+ratio\s*[:=-]?\s*([0-9.]+)', r'agr\s*[:=-]?\s*([0-9.]+)'],

    # General
    'age': [r'\bage\s*[:=-]?\s*([0-9]{1,2})\b', r'patient\s+age\s*[:=-]?\s*([0-9]{1,2})']
}

def parse_lab_text(text):
    """
    Scans raw text using regex patterns and returns parsed parameter values,
    inferred organ, and confidence of organ assignment.
    """
    parsed = {}
    normalized_text = text.lower()

    for param, regex_list in PATTERNS.items():
        for pattern in regex_list:
            match = re.search(pattern, normalized_text)
            if match:
                try:
                    val = float(match.group(1))
                    parsed[param] = val
                    break
                except (ValueError, IndexError):
                    continue

    # Determine target organ based on detected keys
    kidney_keys = {'sc', 'bu', 'sg', 'al', 'bgr', 'wbcc', 'sod', 'pot'}
    liver_keys = {'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase'}

    kidney_score = sum(1 for k in kidney_keys if k in parsed)
    liver_score = sum(1 for k in liver_keys if k in parsed)

    scores = {'kidney': kidney_score, 'liver': liver_score}
    best_organ = max(scores, key=scores.get)

    if scores[best_organ] == 0:
        # Check raw text for organ keywords
        if 'kidney' in normalized_text or 'renal' in normalized_text:
            best_organ = 'kidney'
        elif 'liver' in normalized_text or 'hepatic' in normalized_text:
            best_organ = 'liver'
        else:
            best_organ = 'kidney'

    return {
        'detected_organ': best_organ,
        'organ_scores': scores,
        'parameters': parsed,
        'raw_text_length': len(text)
    }
