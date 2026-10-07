from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from fastapi.testclient import TestClient

from laya_service.app import create_app
from laya_service.config import Settings
from laya_service.db import Database


class FakeEngine:
    def __init__(self, models: tuple[str, ...] = ("laya",)):
        self.loaded = False
        self.default_model = "laya"
        self.models = models

    def load(self, model_id: str | None = None) -> None:
        self.loaded = True

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": model, "name": model, "inputs": ["text"] if model == "laya" else ["text", "image", "audio", "video"]}
            for model in self.models
        ]

    def model_revision(self, model_id: str) -> str | None:
        return None

    def predict(
        self,
        model_id: str,
        state: Any,
        questions: dict[str, Any],
        media: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "answers": {
                name: {"choice": next(iter(question.get("criteria", ["yes"])), "yes"), "confidence": 0.9}
                for name, question in questions.items()
            },
            "received_media": bool(media and any(media.get(key) for key in ("images", "audio", "video"))),
        }


def make_client(tmp_path: Path, models: tuple[str, ...] = ("laya",)) -> TestClient:
    settings = Settings(
        database_path=tmp_path / "service.db",
        admin_key="admin-key-with-enough-characters",
        api_key_pepper="pepper-with-at-least-thirty-two-characters",
        eager_load_model=True,
        enabled_models=models,
        omni_kit_path=tmp_path if "laya-omni" in models else None,
    )
    return TestClient(create_app(settings, FakeEngine(models)))


def bootstrap(client: TestClient, allowed_models: list[str] | None = None) -> tuple[str, str, str]:
    admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
    response = client.post("/admin/users", headers=admin, json={"name": "Example tenant"})
    assert response.status_code == 201
    user_id = response.json()["id"]
    response = client.post(
        f"/admin/users/{user_id}/api-keys",
        headers=admin,
        json={"name": "production", **({"allowed_models": allowed_models} if allowed_models else {})},
    )
    assert response.status_code == 201
    return user_id, response.json()["id"], response.json()["api_key"]


def test_authenticated_prediction_feedback_and_export(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        user_id, _, api_key = bootstrap(client)
        response = client.post(
            "/v1/predict",
            headers={"X-API-Key": api_key},
            json={
                "state": "I was billed twice and need a refund.",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Who should handle this?",
                        "criteria": ["billing", "technical", "sales"],
                    }
                },
                "metadata": {"source": "test"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["result"]["answers"]["department"]["choice"] == "billing"

        response = client.post(
            f"/v1/feedback/{body['request_id']}",
            headers={"X-API-Key": api_key},
            json={"rating": "correct"},
        )
        assert response.status_code == 200

        admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
        response = client.get("/admin/usage", headers=admin, params={"user_id": user_id})
        assert response.json()["items"][0]["succeeded"] == 1

        response = client.get("/admin/dataset/export", headers=admin)
        assert response.status_code == 200
        exported = response.json()
        assert exported["request_id"] == body["request_id"]
        assert exported["feedback"]["rating"] == "correct"
        assert exported["state"] == "I was billed twice and need a refund."


def test_invalid_key_is_rejected_and_logged(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/predict",
            headers={"X-API-Key": "laya_live_unknown.invalid"},
            json={
                "state": "hello",
                "questions": {"safe": {"type": "noul", "instructions": "Is this safe?"}},
            },
        )
        assert response.status_code == 401
        assert "X-Request-ID" in response.headers

        admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
        logs = client.get("/admin/logs", headers=admin).json()["items"]
        assert logs[0]["status"] == "rejected"
        assert logs[0]["api_key_public_id"] == "unknown"


def test_revoked_key_cannot_predict(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        _, key_id, api_key = bootstrap(client)
        admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
        assert client.delete(f"/admin/api-keys/{key_id}", headers=admin).status_code == 204
        response = client.post(
            "/v1/predict",
            headers={"X-API-Key": api_key},
            json={
                "state": "hello",
                "questions": {"safe": {"type": "noul", "instructions": "Is this safe?"}},
            },
        )
        assert response.status_code == 403


def test_validation_failure_has_request_id_and_access_log(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post("/v1/predict", json={"state": "missing questions"})
        assert response.status_code == 422
        assert response.headers["X-Request-ID"]

        admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
        records = client.get("/admin/access-logs", headers=admin).json()["items"]
        failed = next(item for item in records if item["request_id"] == response.headers["X-Request-ID"])
        assert failed["path"] == "/v1/predict"
        assert failed["status_code"] == 422

        response = client.post(
            "/v1/predict",
            json={"state": "invalid question", "questions": {"q": {"type": "choice", "instructions": "Pick"}}},
        )
        assert response.status_code == 422


def test_admin_dashboard_is_served(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/admin")
        assert response.status_code == 200
        assert "Laya MLX Admin" in response.text
        assert "Admin console" in response.text
        assert client.get("/playground").status_code == 200
        assert "Laya Playground" in client.get("/playground").text
        assert client.get("/", follow_redirects=False).headers["location"] == "/playground"


def test_key_model_permissions_and_model_discovery(tmp_path: Path) -> None:
    with make_client(tmp_path, ("laya", "laya-omni")) as client:
        _, key_id, api_key = bootstrap(client, ["laya-omni"])
        headers = {"X-API-Key": api_key}
        assert [item["id"] for item in client.get("/v1/models", headers=headers).json()["data"]] == ["laya-omni"]

        denied = client.post(
            "/v1/predict",
            headers=headers,
            json={"model": "laya", "state": "hello", "questions": {"safe": {"type": "noul", "instructions": "Safe?"}}},
        )
        assert denied.status_code == 403

        allowed = client.post(
            "/v1/predict",
            headers=headers,
            json={"model": "laya-omni", "state": "hello", "questions": {"safe": {"type": "noul", "instructions": "Safe?"}}},
        )
        assert allowed.status_code == 200
        assert allowed.json()["model"] == "laya-omni"

        admin = {"Authorization": "Bearer admin-key-with-enough-characters"}
        changed = client.patch(
            f"/admin/api-keys/{key_id}", headers=admin, json={"allowed_models": ["laya", "laya-omni"]}
        )
        assert changed.status_code == 200
        assert changed.json()["allowed_models"] == ["laya", "laya-omni"]


def test_multimodal_endpoint_passes_uploads_to_selected_model(tmp_path: Path) -> None:
    with make_client(tmp_path, ("laya", "laya-omni")) as client:
        _, _, api_key = bootstrap(client, ["laya-omni"])
        response = client.post(
            "/v1/predict/multimodal",
            headers={"X-API-Key": api_key},
            data={
                "model": "laya-omni",
                "state": "Image.",
                "questions": '{"animal":{"type":"choice","instructions":"What animal?","criteria":["cat","dog"]}}',
            },
            files={"images": ("cat.jpg", b"fake image bytes", "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["result"]["received_media"] is True


def test_existing_database_adds_default_model_permission(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE api_keys (
              id TEXT PRIMARY KEY, public_id TEXT NOT NULL UNIQUE, user_id TEXT NOT NULL,
              name TEXT NOT NULL, secret_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
              expires_at TEXT, last_used_at TEXT, created_at TEXT NOT NULL, revoked_at TEXT
            )
        """)
        connection.execute(
            "INSERT INTO api_keys (id, public_id, user_id, name, secret_hash, created_at) VALUES ('1','pub','user','old','hash','now')"
        )
    Database(path).initialize()
    with sqlite3.connect(path) as connection:
        value = connection.execute(
            "SELECT allowed_models_json FROM api_keys WHERE id = '1'"
        ).fetchone()[0]
    assert value == '["laya"]'
