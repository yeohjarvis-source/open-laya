# Laya MLX API service

An authenticated FastAPI service for Laya text and Laya Omni multimodal models. It serves typed-decision inference on port **6767**, grants model access per API key, includes a public bring-your-own-key playground, accounts for usage, and stores versioned prediction records suitable for later curation into a training dataset.

## Requirements

- Apple Silicon Mac with macOS 14+
- Python 3.11–3.13
- Enough disk space for the selected Hugging Face checkpoint

MLX inference must run natively on macOS. A conventional Linux Docker container is therefore not an appropriate deployment target for this service.

## Install and run

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[mlx]'

export LAYA_ADMIN_KEY="$(openssl rand -hex 24)"
export LAYA_API_KEY_PEPPER="$(openssl rand -hex 32)"
export LAYA_DATABASE_PATH="data/laya-service.db"

laya-service
```

The service listens on `0.0.0.0:6767`. The customer playground is at `http://localhost:6767/playground`, the management dashboard is at `http://localhost:6767/admin`, and OpenAPI documentation is at `http://localhost:6767/docs`. The default model loads during startup. Other enabled models load on their first request so both large runtimes do not consume memory unnecessarily.

### Enable Laya Omni

The service process must use the environment created by `laya-omni-kit`, because that environment contains PyTorch, Pillow, media decoders, and the bundled `laya_omni` wheel:

```bash
cd ../laya-omni-kit
./setup.sh

cd ../laya-mlx
uv pip install --python ../laya-omni-kit/.venv/bin/python -e '.[mlx]'
export LAYA_ENABLED_MODELS=laya,laya-omni
export LAYA_DEFAULT_MODEL=laya
export LAYA_OMNI_KIT_PATH="$(cd ../laya-omni-kit && pwd)"
../laya-omni-kit/.venv/bin/laya-service
```

Use an absolute `LAYA_OMNI_KIT_PATH` in launchd. If only text inference is needed, keep `LAYA_ENABLED_MODELS=laya` and use the smaller service environment described above.
Set `LAYA_SERVICE_VENV` to the absolute `laya-omni-kit/.venv` path when using the included launchd script.

For development without downloading the model:

```bash
uv pip install -e '.[dev]'
pytest
```

## Provision a user and key

Admin endpoints use `Authorization: Bearer $LAYA_ADMIN_KEY`.

```bash
curl -sS http://localhost:6767/admin/users \
  -H "Authorization: Bearer $LAYA_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Acme","external_id":"acme-prod"}'
```

Use the returned user `id`:

```bash
curl -sS http://localhost:6767/admin/users/USER_ID/api-keys \
  -H "Authorization: Bearer $LAYA_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"production","allowed_models":["laya","laya-omni"]}'
```

The full `api_key` is returned only once. The database stores an HMAC-SHA256 digest, not the credential. Rotating `LAYA_API_KEY_PEPPER` invalidates every existing key, so keep it in a secrets manager and back it up separately from the database.

## Predict

```bash
curl -sS http://localhost:6767/v1/predict \
  -H "X-API-Key: $LAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"laya",
    "state":"I was billed twice. Please refund the duplicate.",
    "questions":{
      "department":{
        "type":"choice",
        "instructions":"Who should handle this?",
        "criteria":["billing","technical","sales"]
      },
      "urgent":{
        "type":"noul",
        "instructions":"Does this require urgent attention?"
      }
    },
    "metadata":{"source":"support-api","conversation_id":"conv-123"},
    "dataset_eligible":true
  }'
```

Every accepted or rejected prediction gets a UUID `request_id`. Successful responses have this shape:

```json
{
  "request_id": "...",
  "model": "laya",
  "result": {"answers": {}},
  "usage": {"input_characters": 52, "questions": 2, "latency_ms": 14}
}
```

An authenticated client can discover only the models its key is allowed to use:

```bash
curl -sS http://localhost:6767/v1/models -H "X-API-Key: $LAYA_API_KEY"
```

For images, audio, or video, select `laya-omni` and use the multipart endpoint:

```bash
curl -sS http://localhost:6767/v1/predict/multimodal \
  -H "X-API-Key: $LAYA_API_KEY" \
  -F model=laya-omni \
  -F state=Image. \
  -F 'questions={"animal":{"type":"choice","instructions":"What animal?","criteria":["cat","dog"]}}' \
  -F images=@../laya-omni-kit/samples/cats.jpg
```

The JSON `/v1/predict` endpoint also accepts `model: "laya-omni"` for text-only Omni calls. The multipart endpoint accepts repeated `images` fields (up to eight including video frames), one `audio`, one `video`, `video_frames` from 1–8, and `detail=true|false`.

Submit a later human judgment to turn an inference record into a labeled example:

```bash
curl -sS http://localhost:6767/v1/feedback/REQUEST_ID \
  -H "X-API-Key: $LAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"rating":"incorrect","corrected_answers":{"department":"billing"},"notes":"Reviewer verified"}'
```

## Management API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/users` | Create a user/tenant |
| `GET` | `/admin/users` | List users |
| `PATCH` | `/admin/users/{id}` | Rename, update metadata, or deactivate |
| `POST` | `/admin/users/{id}/api-keys` | Issue a key with optional expiry |
| `GET` | `/admin/users/{id}/api-keys` | List key metadata (never secrets) |
| `PATCH` | `/admin/api-keys/{id}` | Replace the key's allowed-model list |
| `DELETE` | `/admin/api-keys/{id}` | Revoke a key |
| `GET` | `/admin/models` | List models enabled on this service |
| `GET` | `/admin/usage` | Daily success/failure counts and latency |
| `GET` | `/admin/access-logs` | Audit every HTTP request, including validation failures |
| `GET` | `/admin/logs` | Search audit records; payloads are hidden by default |
| `GET` | `/admin/logs/{request_id}` | Inspect a complete record |
| `GET` | `/admin/dataset/export` | Export successful, eligible records as JSONL |

Usage and log endpoints accept time range and user filters. Export is capped by `LAYA_MAX_LOG_EXPORT_ROWS` (default `10000`) to avoid unbounded memory use.

## Dataset and privacy design

The SQLite log ties each authenticated example to the user, key, model ID/revision, schema version, input, question schema, output, timing, caller metadata, eligibility flag, and optional feedback/corrections. This preserves training provenance and avoids mixing authentication secrets into examples. API keys are never logged. Rejected calls retain only audit metadata—not their submitted payload—so anonymous callers cannot fill the dataset store.

Raw inputs can contain personal or regulated data. Before production use:

- document user consent and retention periods;
- set `dataset_eligible: false` for records that must not be trained on;
- use `LAYA_STORE_PAYLOADS=false` if audit metadata is needed but raw content must not persist;
- restrict admin endpoints at the network layer and use TLS at a reverse proxy;
- back up and encrypt the SQLite database, or replace it with a managed database for multi-host deployment;
- periodically export, validate, redact, deduplicate, and split data by user/conversation before training.

SQLite WAL mode is appropriate for a single service process. Run one Uvicorn worker because the MLX model is process-local and each additional worker loads another full copy. For high-volume or multi-host serving, migrate the repository layer to PostgreSQL and use a dedicated inference worker queue.

## Start automatically on macOS login

This repository includes a `launchd` agent that starts the native MLX service after login and restarts it if it crashes. MLX should run as a user LaunchAgent—not a pre-login system daemon—so it has access to the user's Metal session.

```bash
mkdir -p ~/Library/LaunchAgents
cp scripts/com.laya.mlx-service.plist ~/Library/LaunchAgents/
chmod 600 .env ~/Library/LaunchAgents/com.laya.mlx-service.plist
chmod 700 scripts/run-service.sh scripts/laya-service-control
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.laya.mlx-service.plist
```

Manage it afterward with:

```bash
scripts/laya-service-control status
scripts/laya-service-control restart
scripts/laya-service-control stop
scripts/laya-service-control start
scripts/laya-service-control logs
```

The agent writes stdout and stderr to `data/service.stdout.log` and `data/service.stderr.log`. Keep `.env` readable only by your macOS user because it contains the admin key and API-key hashing pepper.

## Configuration

See [`.env.example`](.env.example). Required secrets are deliberately not given working defaults. `LAYA_ENABLED_MODELS` controls the server-wide model catalog, while each key's `allowed_models` is a subset of that catalog. Existing keys are migrated with `laya` access only. `LAYA_MODEL_ID` is the underlying text checkpoint; set `LAYA_MODEL_REVISION` to a fixed upstream revision in production so dataset provenance stays reproducible.
