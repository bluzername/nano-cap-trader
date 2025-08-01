from starlette.testclient import TestClient
from main import app

def test_status_endpoint():
    client = TestClient(app)
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "cash" in r.json()