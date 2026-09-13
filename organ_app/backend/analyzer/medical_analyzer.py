"""
Medical Analyzer: Clinical rule interpretation, biomarker evaluations,
risk categorization, and personalized medical/lifestyle recommendations.
"""

REFERENCE_RANGES = {
    'kidney': {
        'age': {'name': 'Age', 'unit': 'years', 'min': 1, 'max': 120, 'type': 'numeric', 'normal_str': '18–80 years'},
        'sc': {'name': 'Serum Creatinine', 'unit': 'mg/dL', 'min': 0.6, 'max': 1.2, 'critical_high': 3.0, 'type': 'numeric', 'normal_str': '0.6–1.2 mg/dL'},
        'bu': {'name': 'Blood Urea', 'unit': 'mg/dL', 'min': 15.0, 'max': 45.0, 'critical_high': 80.0, 'type': 'numeric', 'normal_str': '15–40 mg/dL'},
        'al': {'name': 'Urine Albumin', 'unit': 'grade', 'min': 0.0, 'max': 0.0, 'critical_high': 3.0, 'type': 'numeric', 'normal_str': '0 (Negative)'},
        'su': {'name': 'Urine Sugar', 'unit': 'grade', 'min': 0.0, 'max': 0.0, 'critical_high': 2.0, 'type': 'numeric', 'normal_str': '0 (Negative)'},
        'rbc': {'name': 'Red Blood Cells', 'unit': '', 'type': 'categorical', 'normal_val': 'normal', 'normal_str': 'Normal'},
        'pc': {'name': 'Pus Cell', 'unit': '', 'type': 'categorical', 'normal_val': 'normal', 'normal_str': 'Normal'},
        'pcc': {'name': 'Pus Cell Clumps', 'unit': '', 'type': 'categorical', 'normal_val': 'notpresent', 'normal_str': 'Not Present'},
        'ba': {'name': 'Bacteria', 'unit': '', 'type': 'categorical', 'normal_val': 'notpresent', 'normal_str': 'Not Present'},
        'bgr': {'name': 'Blood Glucose Random', 'unit': 'mg/dL', 'min': 70.0, 'max': 140.0, 'critical_high': 200.0, 'type': 'numeric', 'normal_str': '70–140 mg/dL'},
        'bp': {'name': 'Blood Pressure', 'unit': 'mmHg', 'min': 60.0, 'max': 80.0, 'critical_high': 100.0, 'type': 'numeric', 'normal_str': '60–80 mmHg'},
        'hemo': {'name': 'Hemoglobin', 'unit': 'g/dL', 'min': 12.0, 'max': 17.5, 'critical_low': 8.0, 'type': 'numeric', 'normal_str': '12–17 g/dL'},
        'pcv': {'name': 'Packed Cell Volume', 'unit': '%', 'min': 40.0, 'max': 52.0, 'type': 'numeric', 'normal_str': '40–52 %'},
        'wbcc': {'name': 'White Blood Cell Count', 'unit': '/uL', 'min': 4500.0, 'max': 11000.0, 'critical_high': 18000.0, 'type': 'numeric', 'normal_str': '4,500–11,000 /uL'},
        'rbcc': {'name': 'Red Blood Cell Count', 'unit': 'mill/cmm', 'min': 4.2, 'max': 6.1, 'type': 'numeric', 'normal_str': '4.2–6.1 mill/cmm'},
        'pot': {'name': 'Serum Potassium', 'unit': 'mEq/L', 'min': 3.5, 'max': 5.1, 'critical_high': 6.0, 'type': 'numeric', 'normal_str': '3.5–5.1 mEq/L'},
        'sod': {'name': 'Serum Sodium', 'unit': 'mEq/L', 'min': 135.0, 'max': 145.0, 'critical_low': 125.0, 'type': 'numeric', 'normal_str': '135–145 mEq/L'},
        'sg': {'name': 'Specific Gravity', 'unit': '', 'min': 1.015, 'max': 1.025, 'critical_low': 1.005, 'type': 'numeric', 'normal_str': '1.015–1.025'}
    },
    'liver': {
        'age': {'name': 'Age', 'unit': 'years', 'min': 18, 'max': 80, 'type': 'numeric', 'normal_str': '18–80 years'},
        'gender': {'name': 'Gender', 'unit': '', 'type': 'info', 'normal_str': 'Biological'},
        'total_bilirubin': {'name': 'Total Bilirubin', 'unit': 'mg/dL', 'min': 0.2, 'max': 1.2, 'critical_high': 3.0, 'type': 'numeric', 'normal_str': '0.2–1.2 mg/dL'},
        'direct_bilirubin': {'name': 'Direct Bilirubin', 'unit': 'mg/dL', 'min': 0.0, 'max': 0.3, 'critical_high': 1.5, 'type': 'numeric', 'normal_str': '0.0–0.3 mg/dL'},
        'alkaline_phosphotase': {'name': 'Alkaline Phosphatase (ALP)', 'unit': 'IU/L', 'min': 44.0, 'max': 147.0, 'critical_high': 350.0, 'type': 'numeric', 'normal_str': '44–147 IU/L'},
        'alamine_aminotransferase': {'name': 'ALT / SGPT', 'unit': 'IU/L', 'min': 7.0, 'max': 56.0, 'critical_high': 150.0, 'type': 'numeric', 'normal_str': '7–56 IU/L'},
        'aspartate_aminotransferase': {'name': 'AST / SGOT', 'unit': 'IU/L', 'min': 10.0, 'max': 40.0, 'critical_high': 150.0, 'type': 'numeric', 'normal_str': '10–40 IU/L'},
        'total_proteins': {'name': 'Total Proteins', 'unit': 'g/dL', 'min': 6.0, 'max': 8.3, 'critical_low': 5.0, 'type': 'numeric', 'normal_str': '6.0–8.3 g/dL'},
        'albumin': {'name': 'Serum Albumin', 'unit': 'g/dL', 'min': 3.5, 'max': 5.0, 'critical_low': 2.5, 'type': 'numeric', 'normal_str': '3.5–5.0 g/dL'},
        'albumin_and_globulin_ratio': {'name': 'Albumin/Globulin Ratio', 'unit': '', 'min': 1.1, 'max': 2.5, 'critical_low': 0.8, 'type': 'numeric', 'normal_str': '1.1–2.5'}
    }
}

def determine_risk_tier(risk_score):
    """
    User-specified risk stratification:
    0–34%  -> Low Risk
    35–64% -> Medium Risk
    65–84% -> High Risk
    85–100% -> Critical Risk
    """
    score = float(risk_score)
    if score >= 85.0:
        return 'Critical Risk'
    elif score >= 65.0:
        return 'High Risk'
    elif score >= 35.0:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def analyze_biomarkers(organ_type, inputs):
    """
    Evaluates every parameter entered or extracted against reference ranges.
    Calculates dynamic status: Normal, High, Low, Abnormal.
    Returns complete parameter list, abnormal list, and normal list.
    """
    organ_key = organ_type.lower()
    ref_dict = REFERENCE_RANGES.get(organ_key, {})

    evaluated_params = []
    abnormal_params = []
    normal_params = []

    for param_key, raw_val in inputs.items():
        if raw_val is None or str(raw_val).strip() == '':
            continue

        ref = ref_dict.get(param_key)
        param_name = ref['name'] if ref else param_key.replace('_', ' ').title()
        unit = ref.get('unit', '') if ref else ''
        normal_range_str = ref.get('normal_str', 'Clinical reference') if ref else 'Clinical'
        p_type = ref.get('type', 'numeric') if ref else 'numeric'

        status = 'Normal'
        css_class = 'badge-normal'

        # Parse numeric if possible
        num_val = None
        try:
            num_val = float(raw_val)
            formatted_val = f"{int(num_val)}" if num_val.is_integer() else f"{round(num_val, 2)}"
        except (ValueError, TypeError):
            formatted_val = str(raw_val).strip()

        if p_type == 'info':
            status = 'Normal'
            css_class = 'badge-normal'
        elif p_type == 'categorical':
            val_lower = str(raw_val).strip().lower()
            expected_norm = ref.get('normal_val', 'normal')
            if val_lower in [expected_norm, '0', 'negative', 'absent', 'notpresent']:
                status = 'Normal'
                css_class = 'badge-normal'
            else:
                status = 'Abnormal'
                css_class = 'badge-critical'
        elif num_val is not None and ref and 'min' in ref and 'max' in ref:
            if 'critical_high' in ref and num_val >= ref['critical_high']:
                status = 'High'
                css_class = 'badge-critical'
            elif 'critical_low' in ref and num_val <= ref['critical_low']:
                status = 'Low'
                css_class = 'badge-critical'
            elif num_val > ref['max']:
                status = 'High'
                css_class = 'badge-warning'
            elif num_val < ref['min']:
                status = 'Low'
                css_class = 'badge-warning'
            else:
                status = 'Normal'
                css_class = 'badge-normal'

        patient_display_val = f"{formatted_val} {unit}".strip()

        param_record = {
            'key': param_key,
            'name': param_name,
            'value': patient_display_val,
            'raw_value': raw_val,
            'normal_range': normal_range_str,
            'status': status,
            'css_class': css_class
        }
        evaluated_params.append(param_record)

        if status in ['High', 'Low', 'Abnormal']:
            abnormal_params.append({
                'name': param_name,
                'value': patient_display_val,
                'status': status,
                'key': param_key
            })
        else:
            normal_params.append({
                'name': param_name,
                'value': patient_display_val,
                'status': 'Normal',
                'key': param_key
            })

    return {
        'parameters': evaluated_params,
        'abnormal_parameters': abnormal_params,
        'normal_parameters': normal_params,
        'abnormal_count': len(abnormal_params),
        'normal_count': len(normal_params)
    }

def generate_clinical_report(organ_type, prediction_dict, inputs):
    organ_key = organ_type.lower()
    risk_score = prediction_dict.get('risk_score', 0.0)
    risk_level = determine_risk_tier(risk_score)
    prediction_dict['risk_level'] = risk_level  # Align with 4 tiers

    bio_eval = analyze_biomarkers(organ_key, inputs)
    abnormal_items = bio_eval['abnormal_parameters']

    abnormal_names = [f"{a['name']} ({a['status']})" for a in abnormal_items]
    summary_text = ""
    recommendations = []

    if organ_key == 'kidney':
        recommendations = [
            "Consult a nephrologist for abnormal kidney results.",
            "Follow professional medical advice regarding renal preservation.",
            "Monitor kidney function as advised with periodic serum creatinine and eGFR testing."
        ]
        if abnormal_items:
            recommendations.append("Restrict dietary sodium (< 2,000 mg/day) and review any nephrotoxic medications (NSAIDs).")
            recommendations.append("Maintain optimal blood pressure target < 130/80 mmHg.")
        if risk_level in ['High Risk', 'Critical Risk']:
            summary_text = (
                f"MultiOrganAI assessment indicates elevated probability of renal disease (Risk: {risk_score}%, {risk_level}). "
                f"Biomarkers flagged: {', '.join(abnormal_names) if abnormal_names else 'Filtration impairment markers'}. "
                "Immediate clinical evaluation by a nephrology specialist is strongly indicated."
            )
        else:
            summary_text = (
                f"Kidney profile is within acceptable clinical tolerances (Risk: {risk_score}%, {risk_level}). "
                "Glomerular filtration and electrolyte balances demonstrate stability."
            )

    elif organ_key == 'liver':
        recommendations = [
            "Consult a hepatologist or physician.",
            "Monitor liver function tests (LFTs) every 4–8 weeks as advised.",
            "Follow medical advice regarding abnormal liver markers, enzyme changes, or steatosis."
        ]
        if abnormal_items:
            recommendations.append("Strictly avoid alcohol, unverified herbal supplements, and excessive acetaminophen.")
            recommendations.append("Schedule an abdominal ultrasound to evaluate hepatic parenchymal structure.")
        if risk_level in ['High Risk', 'Critical Risk']:
            summary_text = (
                f"Hepatic diagnostic analysis reveals marked enzyme elevation or synthetic variance (Risk: {risk_score}%, {risk_level}). "
                f"Elevated markers: {', '.join(abnormal_names) if abnormal_names else 'Hepatic transaminases/bilirubin'}. "
                "Specialist hepatology assessment is advised to investigate underlying etiology."
            )
        else:
            summary_text = (
                f"Hepatic function markers are within normal physiological bounds (Risk: {risk_score}%, {risk_level}). "
                "Bilirubin excretion and enzyme synthesis demonstrate normal clinical function."
            )

    # Medical disclaimer
    disclaimer = (
        "This AI system is designed for screening and educational support. "
        "It is not a replacement for professional medical diagnosis, treatment, or emergency medical care."
    )

    return {
        'summary': summary_text,
        'recommendations': recommendations,
        'disclaimer': disclaimer,
        'parameters': bio_eval['parameters'],
        'abnormal_parameters': bio_eval['abnormal_parameters'],
        'normal_parameters': bio_eval['normal_parameters'],
        'abnormal_count': bio_eval['abnormal_count'],
        'normal_count': bio_eval['normal_count'],
        'risk_level': risk_level,
        'risk_score': risk_score
    }
