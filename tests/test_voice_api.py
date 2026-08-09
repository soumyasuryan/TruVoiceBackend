import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.auth import create_access_token
from app.services.twilio_service import generate_twiml_response

class TestVoiceAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.mock_user_id = "test-user-uuid-1234"
        self.auth_token = create_access_token({"sub": self.mock_user_id})

    def test_unauthenticated_voice_token(self):
        response = self.client.post("/api/v1/voice/token")
        self.assertEqual(response.status_code, 401)

    def test_authenticated_voice_token(self):
        response = self.client.post(
            "/api/v1/voice/token",
            headers={"Authorization": f"Bearer {self.auth_token}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("token", data)
        self.assertEqual(data["identity"], f"user_{self.mock_user_id}")

    def test_outgoing_call_validation(self):
        # Invalid phone number
        response = self.client.post(
            "/api/v1/voice/outgoing",
            json={"phone_number": "123"},
            headers={"Authorization": f"Bearer {self.auth_token}"}
        )
        self.assertIn(response.status_code, [400, 422])

    def test_twiml_generation(self):
        twiml = generate_twiml_response("call-123", "http://localhost:8000")
        self.assertIn("<Response>", twiml)
        self.assertIn("<Stream", twiml)
        self.assertIn("call-123", twiml)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

if __name__ == "__main__":
    unittest.main()
