from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  external_id TEXT UNIQUE,
  active INTEGER NOT NULL DEFAULT 1,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  public_id TEXT NOT NULL UNIQUE,
  user_id TEXT NOT NULL REFERENCES users(id),
  name TEXT NOT NULL,
  secret_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  expires_at TEXT,
  last_used_at TEXT,
  created_at TEXT NOT NULL,
  revoked_at TEXT,
  allowed_models_json TEXT NOT NULL DEFAULT '["laya"]'
);
CREATE TABLE IF NOT EXISTS prediction_logs (
  request_id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES users(id),
  api_key_id TEXT REFERENCES api_keys(id),
  api_key_public_id TEXT,
  received_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL,
  http_status INTEGER,
  model_id TEXT NOT NULL,
  model_revision TEXT,
  state_json TEXT,
  questions_json TEXT,
  response_json TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  dataset_eligible INTEGER NOT NULL DEFAULT 1,
  error_type TEXT,
  error_message TEXT,
  latency_ms REAL,
  client_ip TEXT,
  user_agent TEXT,
  feedback_json TEXT,
  feedback_at TEXT,
  schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_user_time
  ON prediction_logs(user_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_key_time
  ON prediction_logs(api_key_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_dataset
  ON prediction_logs(dataset_eligible, status, received_at DESC);
CREATE TABLE IF NOT EXISTS access_logs (
  request_id TEXT PRIMARY KEY,
  occurred_at TEXT NOT NULL,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  client_ip TEXT,
  user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_access_logs_time ON access_logs(occurred_at DESC);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(api_keys)")}
            if "allowed_models_json" not in columns:
                connection.execute(
                    "ALTER TABLE api_keys ADD COLUMN allowed_models_json TEXT NOT NULL DEFAULT '[\"laya\"]'"
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_user(self, name: str, external_id: str | None, metadata: dict[str, Any]) -> dict[str, Any]:
        now, user_id = utc_now(), str(uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, 1, ?, ?, ?)",
                (user_id, name, external_id, json_dump(metadata), now, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self.user_dict(row)

    def list_users(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [self.user_dict(row) for row in rows]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self.user_dict(row) if row else None

    def patch_user(self, user_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        fields, params = [], []
        for key in ("name", "active"):
            if key in values:
                fields.append(f"{key} = ?")
                params.append(int(values[key]) if key == "active" else values[key])
        if "metadata" in values:
            fields.append("metadata_json = ?")
            params.append(json_dump(values["metadata"]))
        if not fields:
            return self.get_user(user_id)
        fields.append("updated_at = ?")
        params.extend([utc_now(), user_id])
        with self.connect() as conn:
            conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
        return self.get_user(user_id)

    @staticmethod
    def user_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "name": row["name"], "external_id": row["external_id"],
            "active": bool(row["active"]), "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def create_key(
        self,
        user_id: str,
        name: str,
        public_id: str,
        secret_hash: str,
        expires_at: str | None,
        allowed_models: list[str],
    ) -> dict[str, Any]:
        key_id, now = str(uuid4()), utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (id, public_id, user_id, name, secret_hash, expires_at, created_at, allowed_models_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key_id, public_id, user_id, name, secret_hash, expires_at, now, json_dump(allowed_models)),
            )
        return self.get_key(key_id)

    def get_key(self, key_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return self.key_dict(row) if row else None

    def find_key_by_public_id(self, public_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT k.*, u.active AS user_active FROM api_keys k JOIN users u ON u.id = k.user_id WHERE k.public_id = ?",
                (public_id,),
            ).fetchone()

    def list_keys(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
            ).fetchall()
        return [self.key_dict(row) for row in rows]

    @staticmethod
    def key_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = {key: row[key] for key in (
            "id", "public_id", "user_id", "name", "expires_at", "last_used_at", "created_at", "revoked_at"
        )} | {"active": bool(row["active"])}
        result["allowed_models"] = json.loads(row["allowed_models_json"])
        return result

    def update_key_models(self, key_id: str, allowed_models: list[str]) -> dict[str, Any] | None:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET allowed_models_json = ? WHERE id = ?",
                (json_dump(allowed_models), key_id),
            )
        return self.get_key(key_id) if cursor.rowcount else None

    def touch_key(self, key_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (utc_now(), key_id))

    def revoke_key(self, key_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET active = 0, revoked_at = ? WHERE id = ? AND active = 1",
                (utc_now(), key_id),
            )
        return cursor.rowcount > 0

    def insert_log(self, values: dict[str, Any]) -> None:
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO prediction_logs ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def finish_log(self, request_id: str, values: dict[str, Any]) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE prediction_logs SET {assignments} WHERE request_id = ?",
                (*values.values(), request_id),
            )

    def set_feedback(self, request_id: str, user_id: str, feedback: dict[str, Any]) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE prediction_logs SET feedback_json = ?, feedback_at = ? WHERE request_id = ? AND user_id = ? AND status = 'succeeded'",
                (json_dump(feedback), utc_now(), request_id, user_id),
            )
        return cursor.rowcount > 0

    def list_logs(self, filters: dict[str, Any], include_payloads: bool = True) -> list[dict[str, Any]]:
        where, params = [], []
        for key in ("user_id", "status"):
            if filters.get(key) is not None:
                where.append(f"{key} = ?")
                params.append(filters[key])
        if filters.get("dataset_eligible") is not None:
            where.append("dataset_eligible = ?")
            params.append(int(filters["dataset_eligible"]))
        if filters.get("start") is not None:
            where.append("received_at >= ?")
            params.append(filters["start"])
        if filters.get("end") is not None:
            where.append("received_at < ?")
            params.append(filters["end"])
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.extend([filters.get("limit", 100), filters.get("offset", 0)])
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM prediction_logs {clause} ORDER BY received_at DESC LIMIT ? OFFSET ?", params
            ).fetchall()
        return [self.log_dict(row, include_payloads) for row in rows]

    def get_log(self, request_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM prediction_logs WHERE request_id = ?", (request_id,)).fetchone()
        return self.log_dict(row, True) if row else None

    @staticmethod
    def log_dict(row: sqlite3.Row, include_payloads: bool) -> dict[str, Any]:
        result = {key: row[key] for key in (
            "request_id", "user_id", "api_key_id", "api_key_public_id", "received_at", "completed_at",
            "status", "http_status", "model_id", "model_revision", "dataset_eligible", "error_type",
            "error_message", "latency_ms", "client_ip", "user_agent", "feedback_at", "schema_version",
        )}
        result["dataset_eligible"] = bool(result["dataset_eligible"])
        for source, target in (("metadata_json", "metadata"), ("feedback_json", "feedback")):
            result[target] = json.loads(row[source]) if row[source] else None
        if include_payloads:
            for source, target in (("state_json", "state"), ("questions_json", "questions"), ("response_json", "response")):
                result[target] = json.loads(row[source]) if row[source] else None
        return result

    def usage(self, user_id: str | None, start: str | None, end: str | None) -> list[dict[str, Any]]:
        where, params = [], []
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if start:
            where.append("received_at >= ?")
            params.append(start)
        if end:
            where.append("received_at < ?")
            params.append(end)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self.connect() as conn:
            rows = conn.execute(f"""
                SELECT user_id, substr(received_at, 1, 10) AS day,
                       COUNT(*) AS requests,
                       SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
                       SUM(CASE WHEN status != 'succeeded' THEN 1 ELSE 0 END) AS failed,
                       ROUND(AVG(CASE WHEN status = 'succeeded' THEN latency_ms END), 3) AS avg_latency_ms
                FROM prediction_logs {clause}
                GROUP BY user_id, day ORDER BY day DESC, user_id
            """, params).fetchall()
        return [dict(row) for row in rows]

    def insert_access_log(self, values: dict[str, Any]) -> None:
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO access_logs ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def list_access_logs(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM access_logs ORDER BY occurred_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [dict(row) for row in rows]
