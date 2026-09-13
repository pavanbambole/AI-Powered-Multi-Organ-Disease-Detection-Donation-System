import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from database import (
    db, User, PredictionRecord, MedicalReport, Donor, Recipient,
    MatchRecord, UploadedFile, Notification, OrganRequest, OrganMatch
)

class TestDatabaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_user_registration_login_jwt(self):
        """Test User Entity: Register, Login, JWT auth, and /api/auth/me."""
        test_email = 'dr.chen@multiorganai.health'
        # Clean if exists from earlier run
        with app.app_context():
            u = User.query.filter_by(email=test_email).first()
            if u:
                db.session.delete(u)
                db.session.commit()

        # 1. Register
        reg_payload = {
            'username': 'dr_chen_test',
            'email': test_email,
            'password': 'SecurePassword2026!',
            'full_name': 'Dr. Lin Chen',
            'phone': '+1-800-555-4321',
            'blood_group': 'B+',
            'age': 41,
            'gender': 'Female',
            'role': 'doctor'
        }
        res = self.client.post('/api/auth/register', json=reg_payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('token', data)
        self.assertEqual(data['user']['phone'], '+1-800-555-4321')
        token = data['token']

        # 2. Login
        login_res = self.client.post('/api/auth/login', json={
            'identifier': test_email,
            'password': 'SecurePassword2026!'
        })
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.get_json()
        self.assertTrue(login_data['success'])
        self.assertIn('token', login_data)

        # 3. Protected /api/auth/me using Bearer token
        me_res = self.client.get('/api/auth/me', headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.get_json()
        self.assertEqual(me_data['user']['email'], test_email)
        self.assertEqual(me_data['user']['phone'], '+1-800-555-4321')

    def test_02_predictions_crud_and_notifications(self):
        """Test Predictions Entity: Create via API, list, get by ID, delete, and notification trigger."""
        # 1. Login to get token
        login_res = self.client.post('/api/auth/login', json={
            'identifier': 'demo@multiorganai.health',
            'password': 'demo1234'
        })
        token = login_res.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        # 2. Create prediction
        pred_payload = {
            'organ': 'kidney',
            'age': 48, 'bp': 85, 'sg': 1.018, 'al': 1, 'su': 0,
            'bgr': 115, 'bu': 34, 'sc': 1.2, 'sod': 138, 'pot': 4.4, 'hemo': 13.8, 'wbcc': 7400
        }
        create_res = self.client.post('/api/predictions', json=pred_payload, headers=headers)
        self.assertEqual(create_res.status_code, 201)
        data = create_res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        pred_obj = data['prediction']
        pred_id = pred_obj['id']
        self.assertEqual(pred_obj['organ'], 'Kidney')
        self.assertIn('confidence', pred_obj)
        self.assertIn('risk_level', pred_obj)

        # 3. List predictions
        list_res = self.client.get('/api/predictions?organ=kidney', headers=headers)
        self.assertEqual(list_res.status_code, 200)
        list_data = list_res.get_json()
        self.assertGreaterEqual(list_data['total'], 1)

        # 4. Get by ID
        get_res = self.client.get(f'/api/predictions/{pred_id}', headers=headers)
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.get_json()['prediction']['id'], pred_id)

        # 5. Check notification was created
        notif_res = self.client.get('/api/notifications', headers=headers)
        self.assertEqual(notif_res.status_code, 200)
        notifs = notif_res.get_json()['notifications']
        self.assertTrue(any('Kidney AI Analysis' in n['title'] for n in notifs))

        # 6. Delete prediction
        del_res = self.client.delete(f'/api/predictions/{pred_id}', headers=headers)
        self.assertEqual(del_res.status_code, 200)

    def test_03_medical_reports(self):
        """Test Medical Reports Entity: List reports, get report details, and check fields."""
        res = self.client.get('/api/reports')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        if data['reports']:
            rep = data['reports'][0]
            self.assertIn('report_id', rep)
            self.assertIn('organ', rep)
            self.assertIn('parameters', rep)
            self.assertIn('disease_detected', rep)
            self.assertIn('confidence', rep)

            # Get single report detail
            rep_detail = self.client.get(f"/api/report/{rep['report_id']}")
            self.assertEqual(rep_detail.status_code, 200)

    def test_04_donors_crud(self):
        """Test Donors Entity: Full CRUD operations & notification."""
        login_res = self.client.post('/api/auth/login', json={
            'identifier': 'demo@multiorganai.health',
            'password': 'demo1234'
        })
        token = login_res.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        # 1. Create Donor
        donor_payload = {
            'full_name': 'Test Donor Arthur',
            'age': 33,
            'gender': 'Male',
            'blood_group': 'O+',
            'organ': 'Kidney',
            'hospital': 'Mount Sinai Hospital, NY',
            'contact_phone': '+1-555-0987',
            'medical_status': 'Cleared',
            'availability': 'Available'
        }
        res = self.client.post('/api/donors', json=donor_payload, headers=headers)
        self.assertEqual(res.status_code, 201)
        donor = res.get_json()['donor']
        donor_id = donor['id']
        self.assertEqual(donor['full_name'], 'Test Donor Arthur')
        self.assertEqual(donor['organ'], 'Kidney')
        self.assertEqual(donor['medical_status'], 'Cleared')

        # 2. List & Filter Donors
        list_res = self.client.get('/api/donors?organ=Kidney&blood_group=O+')
        self.assertEqual(list_res.status_code, 200)
        donors = list_res.get_json()['donors']
        self.assertTrue(any(d['id'] == donor_id for d in donors))

        # 3. Get Single Donor
        get_res = self.client.get(f'/api/donors/{donor_id}')
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.get_json()['donor']['id'], donor_id)

        # 4. Update Donor
        update_res = self.client.put(f'/api/donors/{donor_id}', json={
            'availability': 'Matched',
            'hospital': 'Johns Hopkins Hospital'
        })
        self.assertEqual(update_res.status_code, 200)
        up_donor = update_res.get_json()['donor']
        self.assertEqual(up_donor['availability'], 'Matched')
        self.assertEqual(up_donor['hospital'], 'Johns Hopkins Hospital')

        # 5. Delete Donor
        del_res = self.client.delete(f'/api/donors/{donor_id}')
        self.assertEqual(del_res.status_code, 200)

    def test_05_organ_requests_crud(self):
        """Test Organ Requests Entity: Full CRUD operations & notification."""
        login_res = self.client.post('/api/auth/login', json={
            'identifier': 'demo@multiorganai.health',
            'password': 'demo1234'
        })
        token = login_res.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        # 1. Create Organ Request
        req_payload = {
            'patient_name': 'Test Recipient Clara',
            'age': 50,
            'gender': 'Female',
            'blood_group': 'A+',
            'required_organ': 'Liver',
            'urgency': 'Critical',
            'hospital': 'NYU Langone Health',
            'contact_phone': '+1-555-7766'
        }
        res = self.client.post('/api/organ-requests', json=req_payload, headers=headers)
        self.assertEqual(res.status_code, 201)
        req_obj = res.get_json()['request']
        req_id = req_obj['id']
        self.assertEqual(req_obj['patient_name'], 'Test Recipient Clara')
        self.assertEqual(req_obj['required_organ'], 'Liver')
        self.assertEqual(req_obj['urgency'], 'Critical')

        # 2. List & Filter Organ Requests
        list_res = self.client.get('/api/organ-requests?organ=Liver&urgency=Critical')
        self.assertEqual(list_res.status_code, 200)
        requests = list_res.get_json()['requests']
        self.assertTrue(any(r['id'] == req_id for r in requests))

        # 3. Get Single Request
        get_res = self.client.get(f'/api/organ-requests/{req_id}')
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.get_json()['request']['id'], req_id)

        # 4. Update Request
        up_res = self.client.put(f'/api/organ-requests/{req_id}', json={
            'urgency': 'High',
            'status': 'Matching In Progress'
        })
        self.assertEqual(up_res.status_code, 200)
        self.assertEqual(up_res.get_json()['request']['urgency'], 'High')
        self.assertEqual(up_res.get_json()['request']['status'], 'Matching In Progress')

        # 5. Delete Request
        del_res = self.client.delete(f'/api/organ-requests/{req_id}')
        self.assertEqual(del_res.status_code, 200)

    def test_06_organ_matches_and_algorithm(self):
        """Test Organ Matches Entity: Run matching algorithm, retrieve matches from DB."""
        run_res = self.client.post('/api/organ-matches/run')
        self.assertEqual(run_res.status_code, 201)
        data = run_res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('matches', data)

        # List matches
        list_res = self.client.get('/api/organ-matches')
        self.assertEqual(list_res.status_code, 200)
        matches = list_res.get_json()['matches']
        self.assertGreater(len(matches), 0)
        m = matches[0]
        self.assertIn('donor', m)
        self.assertIn('recipient', m)
        self.assertIn('match_score', m)
        self.assertIn('blood_compatibility', m)

    def test_07_notifications_management(self):
        """Test Notifications Entity: List, mark read, delete."""
        login_res = self.client.post('/api/auth/login', json={
            'identifier': 'demo@multiorganai.health',
            'password': 'demo1234'
        })
        token = login_res.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        res = self.client.get('/api/notifications', headers=headers)
        self.assertEqual(res.status_code, 200)
        notifs = res.get_json()['notifications']
        self.assertGreater(len(notifs), 0)

        notif_id = notifs[0]['id']
        # Mark read
        read_res = self.client.put(f'/api/notifications/{notif_id}/read', headers=headers)
        self.assertEqual(read_res.status_code, 200)
        self.assertTrue(read_res.get_json()['notification']['is_read'])

        # Mark all read
        mark_all_res = self.client.post('/api/notifications/mark-all-read', headers=headers)
        self.assertEqual(mark_all_res.status_code, 200)

    def test_08_uploaded_files_query(self):
        """Test Uploaded Files Entity."""
        res = self.client.get('/api/files')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('files', data)

if __name__ == '__main__':
    unittest.main()
