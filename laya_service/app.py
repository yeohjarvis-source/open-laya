from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from .config import Settings
from .db import Database, json_dump, utc_now
from .engine import LayaEngine, PredictionEngine
from .schemas import ApiKeyCreate, ApiKeyPatch, FeedbackRequest, PredictRequest, PredictResponse, UserCreate, UserPatch
from .security import generate_api_key, hash_api_secret, parse_api_key, secure_equal

logger = logging.getLogger("laya_service")
ADMIN_HTML = Path(__file__).with_name("static").joinpath("admin.html")
PLAYGROUND_HTML = Path(__file__).with_name("static").joinpath("playground.html")


def create_app(settings: Settings | None = None, engine: PredictionEngine | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    database = Database(resolved_settings.database_path)
    prediction_engine = engine or LayaEngine(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        if resolved_settings.eager_load_model:
            await asyncio.to_thread(prediction_engine.load, resolved_settings.default_model)
        yield

    app = FastAPI(
        title="Laya MLX Service",
        version="0.1.0",
        description="Authenticated typed-decision inference and dataset-grade audit logging.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.database = database
    app.state.engine = prediction_engine

    @app.middleware("http")
    async def audit_http_request(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            try:
                database.insert_access_log({
                    "request_id": request_id,
                    "occurred_at": utc_now(),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "client_ip": client_ip(request),
                    "user_agent": request.headers.get("user-agent"),
                })
            except Exception:
                logger.exception("Failed to write access log request_id=%s", request_id)

    def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {resolved_settings.admin_key}"
        if authorization is None or not secure_equal(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid admin credentials")

    def authenticate_api_key(value: str | None) -> tuple[dict[str, Any], str | None]:
        if not value:
            raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        parsed = parse_api_key(value)
        public_id = parsed[0] if parsed else None
        if not parsed:
            raise HTTPException(status_code=401, detail="Invalid API key")
        public_id, secret = parsed
        row = database.find_key_by_public_id(public_id)
        if row is None or not secure_equal(row["secret_hash"], hash_api_secret(secret, resolved_settings.api_key_pepper)):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not row["active"] or not row["user_active"]:
            raise HTTPException(status_code=403, detail="API key or user is inactive")
        if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            raise HTTPException(status_code=403, detail="API key has expired")
        database.touch_key(row["id"])
        key = dict(row)
        key["allowed_models"] = json.loads(row["allowed_models_json"])
        return key, public_id

    def client_ip(request: Request) -> str | None:
        if resolved_settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",", 1)[0].strip()
        return request.client.host if request.client else None

    @app.get("/healthz", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/playground")

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    def playground() -> HTMLResponse:
        return HTMLResponse(
            PLAYGROUND_HTML.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_dashboard() -> HTMLResponse:
        return HTMLResponse(
            ADMIN_HTML.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    def model_catalog() -> dict[str, dict[str, Any]]:
        return {item["id"]: item for item in prediction_engine.list_models()}

    def validate_model_list(models: list[str] | None) -> list[str]:
        selected = models if models is not None else [resolved_settings.default_model]
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise HTTPException(status_code=422, detail="allowed_models must not be empty")
        unknown = set(selected) - set(model_catalog())
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"Unknown or disabled model(s): {', '.join(sorted(unknown))}"
            )
        return selected

    @app.get("/v1/models", tags=["inference"])
    def list_available_models(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> dict[str, Any]:
        key, _ = authenticate_api_key(x_api_key)
        catalog = model_catalog()
        return {"data": [catalog[item] for item in key["allowed_models"] if item in catalog]}

    async def execute_prediction(
        payload: PredictRequest,
        request: Request,
        x_api_key: str | None,
        media: dict[str, Any] | None = None,
    ) -> PredictResponse:
        request_id = request.state.request_id
        started = time.perf_counter()
        received_at = utc_now()
        parsed = parse_api_key(x_api_key or "")
        public_id = parsed[0] if parsed else None
        base_log = {
            "request_id": request_id,
            "api_key_public_id": public_id,
            "received_at": received_at,
            "status": "received",
            "model_id": payload.model,
            "model_revision": prediction_engine.model_revision(payload.model) if payload.model in model_catalog() else None,
            # Payloads are attached only after authentication. Rejected callers get
            # an audit record without being able to fill the training-data store.
            "state_json": None,
            "questions_json": None,
            "metadata_json": "{}",
            "dataset_eligible": 0,
            "client_ip": client_ip(request),
            "user_agent": request.headers.get("user-agent"),
        }
        database.insert_log(base_log)

        try:
            key, _ = authenticate_api_key(x_api_key)
        except HTTPException as exc:
            database.finish_log(request_id, {
                "completed_at": utc_now(), "status": "rejected", "http_status": exc.status_code,
                "error_type": "authentication_error", "error_message": str(exc.detail),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            raise

        if payload.model not in model_catalog():
            database.finish_log(request_id, {
                "user_id": key["user_id"], "api_key_id": key["id"], "completed_at": utc_now(),
                "status": "rejected", "http_status": 404, "error_type": "unknown_model",
                "error_message": f"Unknown or disabled model: {payload.model}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            raise HTTPException(status_code=404, detail="Model not found")
        if payload.model not in key["allowed_models"]:
            database.finish_log(request_id, {
                "user_id": key["user_id"], "api_key_id": key["id"], "completed_at": utc_now(),
                "status": "rejected", "http_status": 403, "error_type": "model_access_denied",
                "error_message": f"API key cannot access model: {payload.model}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            raise HTTPException(status_code=403, detail="API key is not allowed to use this model")

        database.finish_log(request_id, {
            "user_id": key["user_id"],
            "api_key_id": key["id"],
            "state_json": json_dump(payload.state) if resolved_settings.store_payloads else None,
            "questions_json": json_dump({k: v.model_dump(mode="json") for k, v in payload.questions.items()}) if resolved_settings.store_payloads else None,
            "metadata_json": json_dump(payload.metadata),
            "dataset_eligible": int(payload.dataset_eligible and resolved_settings.store_payloads),
        })
        questions = {name: question.model_dump(exclude_none=True) for name, question in payload.questions.items()}
        try:
            result = await asyncio.to_thread(
                prediction_engine.predict, payload.model, payload.state, questions, media
            )
        except ValueError as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            database.finish_log(request_id, {
                "completed_at": utc_now(), "status": "failed", "http_status": 400,
                "error_type": type(exc).__name__, "error_message": str(exc)[:2000], "latency_ms": latency,
            })
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            logger.exception("Prediction failed request_id=%s", request_id)
            database.finish_log(request_id, {
                "completed_at": utc_now(), "status": "failed", "http_status": 500,
                "error_type": type(exc).__name__, "error_message": str(exc)[:2000], "latency_ms": latency,
            })
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

        latency = round((time.perf_counter() - started) * 1000, 3)
        response = PredictResponse(
            request_id=request_id,
            model=payload.model,
            result=result,
            usage={
                "input_characters": len(json_dump(payload.state)),
                "questions": len(payload.questions),
                "latency_ms": round(latency),
            },
        )
        database.finish_log(request_id, {
            "completed_at": utc_now(), "status": "succeeded", "http_status": 200,
            "response_json": json_dump(result) if resolved_settings.store_payloads else None,
            "latency_ms": latency,
        })
        return response

    @app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(
        payload: PredictRequest,
        request: Request,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> PredictResponse:
        return await execute_prediction(payload, request, x_api_key)

    async def read_upload(upload: UploadFile | None) -> dict[str, Any] | None:
        if upload is None or not upload.filename:
            return None
        data = await upload.read()
        if len(data) > resolved_settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename}: larger than {resolved_settings.max_upload_mb:g} MB",
            )
        return {"filename": upload.filename, "content_type": upload.content_type, "data": data}

    @app.post("/v1/predict/multimodal", response_model=PredictResponse, tags=["inference"])
    async def predict_multimodal(
        request: Request,
        questions: Annotated[str, Form()],
        model: Annotated[str, Form()] = "laya-omni",
        state: Annotated[str, Form()] = "",
        metadata: Annotated[str, Form()] = "{}",
        dataset_eligible: Annotated[bool, Form()] = True,
        images: list[UploadFile] = File(default=[]),
        audio: UploadFile | None = File(default=None),
        video: UploadFile | None = File(default=None),
        video_frames: Annotated[int, Form(ge=1, le=8)] = 4,
        detail: Annotated[bool, Form()] = False,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> PredictResponse:
        # Reject bad credentials before reading upload bodies into application memory.
        upload_key, _ = authenticate_api_key(x_api_key)
        if model not in model_catalog():
            raise HTTPException(status_code=404, detail="Model not found")
        if model not in upload_key["allowed_models"]:
            raise HTTPException(status_code=403, detail="API key is not allowed to use this model")
        if len(images) > 8:
            raise HTTPException(status_code=400, detail="At most 8 images are allowed")
        try:
            question_data = json.loads(questions)
            metadata_data = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON form field: {exc.msg}") from exc
        try:
            payload = PredictRequest.model_validate({
                "model": model,
                "state": state,
                "questions": question_data,
                "metadata": metadata_data,
                "dataset_eligible": dataset_eligible,
            })
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        uploaded_images = [item for item in [await read_upload(image) for image in images] if item]
        media = {
            "images": uploaded_images,
            "audio": await read_upload(audio),
            "video": await read_upload(video),
            "video_frames": video_frames,
            "detail": detail,
        }
        return await execute_prediction(payload, request, x_api_key, media)

    @app.post("/v1/feedback/{request_id}", tags=["inference"])
    def feedback(
        request_id: str,
        payload: FeedbackRequest,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> dict[str, Any]:
        key, _ = authenticate_api_key(x_api_key)
        updated = database.set_feedback(request_id, key["user_id"], payload.model_dump(exclude_none=True))
        if not updated:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return {"request_id": request_id, "feedback_saved": True}

    admin = [Depends(require_admin)]

    @app.get("/admin/models", dependencies=admin, tags=["admin"])
    def list_admin_models() -> dict[str, Any]:
        return {"data": prediction_engine.list_models(), "default": resolved_settings.default_model}

    @app.post("/admin/users", status_code=201, dependencies=admin, tags=["admin"])
    def create_user(payload: UserCreate) -> dict[str, Any]:
        try:
            return database.create_user(payload.name, payload.external_id, payload.metadata)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="external_id already exists") from exc

    @app.get("/admin/users", dependencies=admin, tags=["admin"])
    def list_users() -> list[dict[str, Any]]:
        return database.list_users()

    @app.patch("/admin/users/{user_id}", dependencies=admin, tags=["admin"])
    def patch_user(user_id: str, payload: UserPatch) -> dict[str, Any]:
        user = database.patch_user(user_id, payload.model_dump(exclude_unset=True))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    @app.post("/admin/users/{user_id}/api-keys", status_code=201, dependencies=admin, tags=["admin"])
    def create_key(user_id: str, payload: ApiKeyCreate) -> dict[str, Any]:
        user = database.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        expires_at = payload.expires_at
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at and expires_at <= datetime.now(timezone.utc):
            raise HTTPException(status_code=422, detail="expires_at must be in the future")
        complete_key, public_id, secret = generate_api_key()
        allowed_models = validate_model_list(payload.allowed_models)
        record = database.create_key(
            user_id, payload.name, public_id,
            hash_api_secret(secret, resolved_settings.api_key_pepper),
            expires_at.isoformat() if expires_at else None,
            allowed_models,
        )
        return {**record, "api_key": complete_key}

    @app.get("/admin/users/{user_id}/api-keys", dependencies=admin, tags=["admin"])
    def list_keys(user_id: str) -> list[dict[str, Any]]:
        if not database.get_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return database.list_keys(user_id)

    @app.patch("/admin/api-keys/{key_id}", dependencies=admin, tags=["admin"])
    def patch_key(key_id: str, payload: ApiKeyPatch) -> dict[str, Any]:
        key = database.update_key_models(key_id, validate_model_list(payload.allowed_models))
        if not key:
            raise HTTPException(status_code=404, detail="API key not found")
        return key

    @app.delete("/admin/api-keys/{key_id}", status_code=204, dependencies=admin, tags=["admin"])
    def revoke_key(key_id: str) -> None:
        if not database.revoke_key(key_id):
            raise HTTPException(status_code=404, detail="Active API key not found")

    @app.get("/admin/usage", dependencies=admin, tags=["admin"])
    def usage(
        user_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        rows = database.usage(
            user_id,
            start.isoformat() if start else None,
            end.isoformat() if end else None,
        )
        return {"items": rows}

    @app.get("/admin/access-logs", dependencies=admin, tags=["admin"])
    def access_logs(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return {"items": database.list_access_logs(limit, offset), "limit": limit, "offset": offset}

    def log_filters(
        user_id: str | None,
        log_status: str | None,
        dataset_eligible: bool | None,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return {
            "user_id": user_id, "status": log_status, "dataset_eligible": dataset_eligible,
            "start": start.isoformat() if start else None, "end": end.isoformat() if end else None,
            "limit": limit, "offset": offset,
        }

    @app.get("/admin/logs", dependencies=admin, tags=["admin"])
    def list_logs(
        user_id: str | None = None,
        log_status: Annotated[str | None, Query(alias="status")] = None,
        dataset_eligible: bool | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        include_payloads: bool = False,
    ) -> dict[str, Any]:
        filters = log_filters(user_id, log_status, dataset_eligible, start, end, limit, offset)
        return {"items": database.list_logs(filters, include_payloads), "limit": limit, "offset": offset}

    @app.get("/admin/logs/{request_id}", dependencies=admin, tags=["admin"])
    def get_log(request_id: str) -> dict[str, Any]:
        record = database.get_log(request_id)
        if not record:
            raise HTTPException(status_code=404, detail="Log not found")
        return record

    @app.get("/admin/dataset/export", dependencies=admin, tags=["admin"])
    def export_dataset(
        user_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: Annotated[int, Query(ge=1)] = 10_000,
    ) -> StreamingResponse:
        effective_limit = min(limit, resolved_settings.max_log_export_rows)
        filters = log_filters(user_id, "succeeded", True, start, end, effective_limit, 0)
        records = database.list_logs(filters, include_payloads=True)

        def rows():
            for record in reversed(records):
                yield json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

        return StreamingResponse(
            rows(), media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="laya-dataset.jsonl"'},
        )

    return app
