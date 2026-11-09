from fastapi.testclient import TestClient

from app.main import create_app


def test_app_responds() -> None:
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/openapi.json").status_code == 200
