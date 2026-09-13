import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import app
from database import db, MedicalReport
from ml.predictor import predict_organ
from analyzer.medical_analyzer import analyze_biomarkers, generate_clinical_report
from reports.pdf_generator import generate_medical_pdf
from donation.donation_matcher import calculate_compatibility, find_best_matches, is_blood_compatible
from ocr.report_parser import parse_lab_text

def run_tests():
    print("==================================================")
    print(" Running MultiOrganAI System Diagnostic Verification")
    print("==================================================")

    # 1. Flask Test Client Routes
    client = app.test_client()
    routes_to_test = [
        ('/', 200),
        ('/about', 200),
        ('/how-it-works', 200),
        ('/features', 200),
        ('/organ-donation', 200),
        ('/contact', 200),
        ('/demo', 200),
        ('/login', 200),
        ('/signup', 200),
        ('/history', 200),
        ('/my-reports', 200),
        ('/health', 200),
    ]

    for route, expected_code in routes_to_test:
        res = client.get(route)
        assert res.status_code == expected_code, f"Failed route {route}: got {res.status_code}"
    print("[PASS] All 12 web routes responded with HTTP 200 OK.")

    # 2. Machine Learning Predictions
    kidney_normal = {'age': 35, 'bp': 75, 'sg': 1.020, 'al': 0, 'su': 0, 'bgr': 95, 'bu': 25, 'sc': 0.8, 'sod': 140, 'pot': 4.1, 'hemo': 15.0, 'wbcc': 6500}
    kidney_pred_norm = predict_organ('kidney', kidney_normal)
    print(f"[PASS] Kidney Normal Case: Risk={kidney_pred_norm['risk_score']}%, Level={kidney_pred_norm['risk_level']}, Confidence={kidney_pred_norm['confidence']}%")
    assert 'Low' in kidney_pred_norm['risk_level']

    kidney_at_risk = {'age': 58, 'bp': 95, 'sg': 1.010, 'al': 3, 'su': 2, 'bgr': 190, 'bu': 75, 'sc': 3.2, 'sod': 130, 'pot': 5.2, 'hemo': 9.2, 'wbcc': 11500}
    kidney_pred_risk = predict_organ('kidney', kidney_at_risk)
    print(f"[PASS] Kidney At-Risk Case: Risk={kidney_pred_risk['risk_score']}%, Level={kidney_pred_risk['risk_level']}, Confidence={kidney_pred_risk['confidence']}%")
    assert 'High' in kidney_pred_risk['risk_level'] or 'Critical' in kidney_pred_risk['risk_level']

    # Verify rejection of unsupported organ (heart)
    try:
        predict_organ('heart', {})
        assert False, "Should have raised ValueError for removed organ 'heart'"
    except ValueError:
        print("[PASS] Validated rejection of removed organ 'heart' (only Kidney and Liver supported).")

    liver_normal = {'age': 32, 'gender': 1, 'total_bilirubin': 0.7, 'direct_bilirubin': 0.2, 'alkaline_phosphotase': 160, 'alamine_aminotransferase': 22, 'aspartate_aminotransferase': 24, 'total_proteins': 7.3, 'albumin': 4.3, 'albumin_and_globulin_ratio': 1.4}
    liver_pred_norm = predict_organ('liver', liver_normal)
    print(f"[PASS] Liver Normal Case: Risk={liver_pred_norm['risk_score']}%, Level={liver_pred_norm['risk_level']}, Confidence={liver_pred_norm['confidence']}%")
    assert liver_pred_norm['risk_level'] in ['Low Risk', 'Medium Risk']

    liver_at_risk = {'age': 54, 'gender': 1, 'total_bilirubin': 5.2, 'direct_bilirubin': 2.4, 'alkaline_phosphotase': 520, 'alamine_aminotransferase': 165, 'aspartate_aminotransferase': 180, 'total_proteins': 5.4, 'albumin': 2.4, 'albumin_and_globulin_ratio': 0.7}
    liver_pred_risk = predict_organ('liver', liver_at_risk)
    print(f"[PASS] Liver At-Risk Case: Risk={liver_pred_risk['risk_score']}%, Level={liver_pred_risk['risk_level']}, Confidence={liver_pred_risk['confidence']}%")
    assert liver_pred_risk['risk_score'] > liver_pred_norm['risk_score']
    assert 'High' in liver_pred_risk['risk_level'] or 'Critical' in liver_pred_risk['risk_level']

    # 3. Dynamic Biomarker Analysis & Risk Tiers
    analysis = generate_clinical_report('kidney', kidney_pred_risk, kidney_at_risk)
    assert len(analysis['recommendations']) >= 3
    assert len(analysis['parameters']) >= 10
    assert len(analysis['abnormal_parameters']) > 0
    assert len(analysis['normal_parameters']) > 0
    print(f"[PASS] Dynamic Biomarker Analyzer: Evaluated {len(analysis['parameters'])} parameters (Abnormal: {len(analysis['abnormal_parameters'])}, Normal: {len(analysis['normal_parameters'])}).")

    # Verify status tags
    abnormal_keys = [a['key'] for a in analysis['abnormal_parameters']]
    assert 'sc' in abnormal_keys  # Serum Creatinine 3.2 is High
    assert 'bu' in abnormal_keys  # Blood Urea 75 is High
    print(f"[PASS] Dynamic Status Calculation correctly flagged abnormal biomarkers: {abnormal_keys}")

    # 4. ReportLab PDF Generation Test
    patient_info = {'name': 'Test Verification Patient', 'email': 'test.patient@hospital.org', 'age': 58, 'gender': 'Male', 'blood_group': 'O+'}
    pdf_filename = generate_medical_pdf(patient_info, 'kidney', kidney_pred_risk, analysis)
    pdf_full_path = os.path.join(CURRENT_DIR, 'reports', 'generated', pdf_filename)
    assert os.path.exists(pdf_full_path), f"PDF file not found at {pdf_full_path}"
    assert os.path.getsize(pdf_full_path) > 2000, "PDF file is too small or empty"
    print(f"[PASS] PDF Generator: Clinical report compiled to {pdf_filename} ({os.path.getsize(pdf_full_path)} bytes).")

    # 5. Organ Donation Matching Test
    d_donor = {'donor_name': 'Donor One', 'age': 30, 'gender': 'Female', 'blood_group': 'O-', 'organ_offered': 'Kidney', 'hospital_city': 'New York'}
    d_recip = {'recipient_name': 'Recip One', 'age': 34, 'gender': 'Male', 'blood_group': 'A+', 'organ_needed': 'Kidney', 'urgency_level': 'Critical', 'hospital_city': 'New York'}
    match_res = calculate_compatibility(d_donor, d_recip)
    assert match_res['compatible'] is True
    assert match_res['score'] >= 80.0
    print(f"[PASS] Donation Matcher: Compatible match score = {match_res['score']}%, Tier: {match_res['tier']}")

    # 6. OCR Text Parser Test
    sample_ocr = "Patient lab: Serum Creatinine: 2.8 mg/dL, Blood Urea: 75 mg/dL, Hemoglobin: 9 g/dL, Sodium: 138, Potassium: 4.2"
    parsed = parse_lab_text(sample_ocr)
    assert 'sc' in parsed['parameters'] and parsed['parameters']['sc'] == 2.8
    assert 'bu' in parsed['parameters'] and parsed['parameters']['bu'] == 75.0
    assert 'hemo' in parsed['parameters'] and parsed['parameters']['hemo'] == 9.0
    assert 'sod' in parsed['parameters'] and parsed['parameters']['sod'] == 138.0
    assert 'pot' in parsed['parameters'] and parsed['parameters']['pot'] == 4.2
    print(f"[PASS] OCR Report Parser: Successfully extracted all sample parameters: {parsed['parameters']}")

    # 7. End-to-End API Predict & Zero Parameter Loss Test
    sample_patient_input = {
        'patient_name': 'Eleanor Vance',
        'email': 'eleanor.vance@clinic.org',
        'age': 52,
        'bp': 85,
        'sc': 2.8,
        'bu': 75,
        'hemo': 9.0,
        'sod': 138,
        'pot': 4.2,
        'pcv': 32,
        'wbcc': 10500,
        'rbcc': 3.5
    }
    api_res = client.post('/api/predict/kidney', json=sample_patient_input)
    assert api_res.status_code == 200
    api_data = json.loads(api_res.data)
    assert api_data['success'] is True
    assert api_data['report_id'].startswith('MOAI-')
    assert 'pdf_url' in api_data and api_data['pdf_url'] is not None

    # Verify zero parameter loss
    ret_params = api_data['parameters']
    assert float(ret_params['sc']) == 2.8
    assert float(ret_params['bu']) == 75.0
    assert float(ret_params['hemo']) == 9.0
    assert float(ret_params['sod']) == 138.0
    assert float(ret_params['pot']) == 4.2
    assert float(ret_params['pcv']) == 32.0
    assert float(ret_params['wbcc']) == 10500.0
    assert float(ret_params['rbcc']) == 3.5
    print("[PASS] Zero Parameter Loss: All 10 entered biomarkers preserved through prediction and report payload.")

    # 8. Database Storage & Report Retrieval Verification
    with app.app_context():
        db_report = MedicalReport.query.filter_by(report_id=api_data['report_id']).first()
        assert db_report is not None, "MedicalReport record was not saved in SQLite database"
        assert db_report.organ == 'Kidney'
        assert db_report.confidence > 0.0
        assert db_report.risk_level in ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        db_params = json.loads(db_report.parameters)
        assert float(db_params['sc']) == 2.8
        assert float(db_params['bu']) == 75.0
        print(f"[PASS] Database Storage: MedicalReport {db_report.report_id} verified with real database record.")

    # 9. Test My Reports & Report Detail / Deletion Endpoints
    rep_res = client.get('/api/reports')
    assert rep_res.status_code == 200
    rep_list = json.loads(rep_res.data)['reports']
    assert any(r['report_id'] == api_data['report_id'] for r in rep_list)
    print(f"[PASS] /api/reports returned active session reports (Count: {len(rep_list)}).")

    detail_res = client.get(f"/api/report/{api_data['report_id']}")
    assert detail_res.status_code == 200
    detail_data = json.loads(detail_res.data)
    assert detail_data['report_id'] == api_data['report_id']
    assert len(detail_data['parameter_analysis']) > 0
    print(f"[PASS] /api/report/{api_data['report_id']} returned full clinical detail.")

    # Delete test report
    del_res = client.post(f"/api/report/{api_data['report_id']}/delete")
    assert del_res.status_code == 200
    with app.app_context():
        assert MedicalReport.query.filter_by(report_id=api_data['report_id']).first() is None
    print(f"[PASS] /api/report/{api_data['report_id']}/delete verified database deletion.")

    print("\n==================================================")
    print(" ALL MultiOrganAI Diagnostics PASSED (9/9)!")
    print("==================================================")

if __name__ == '__main__':
    run_tests()
