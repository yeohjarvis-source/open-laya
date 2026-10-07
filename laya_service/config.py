from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    admin_key: str
    api_key_pepper: str
    model_id: str = "aac6fef/laya-mlx"
    model_revision: str | None = None
    model_dtype: str = "float16"
    eager_load_model: bool = True
    store_payloads: bool = True
    trust_proxy_headers: bool = False
    max_log_export_rows: int = 10_000
    enabled_models: tuple[str, ...] = ("laya",)
    default_model: str = "laya"
    omni_kit_path: Path | None = None
    max_upload_mb: float = 50.0

    def __post_init__(self) -> None:
        supported = {"laya", "laya-omni"}
        unknown = set(self.enabled_models) - supported
        if unknown:
            raise ValueError(f"Unsupported model(s): {', '.join(sorted(unknown))}")
        if not self.enabled_models:
            raise ValueError("At least one model must be enabled")
        if self.default_model not in self.enabled_models:
            raise ValueError("default_model must be included in enabled_models")
        if "laya-omni" in self.enabled_models and self.omni_kit_path is None:
            raise ValueError("omni_kit_path is required when laya-omni is enabled")

    @classmethod
    def from_env(cls) -> "Settings":
        admin_key = os.getenv("LAYA_ADMIN_KEY", "")
        pepper = os.getenv("LAYA_API_KEY_PEPPER", "")
        if len(admin_key) < 24:
            raise RuntimeError("LAYA_ADMIN_KEY must be at least 24 characters")
        if len(pepper) < 32:
            raise RuntimeError("LAYA_API_KEY_PEPPER must be at least 32 characters")
        revision = os.getenv("LAYA_MODEL_REVISION") or None
        enabled_models = tuple(
            item.strip() for item in os.getenv("LAYA_ENABLED_MODELS", "laya").split(",") if item.strip()
        )
        omni_path = os.getenv("LAYA_OMNI_KIT_PATH")
        return cls(
            database_path=Path(os.getenv("LAYA_DATABASE_PATH", "data/laya-service.db")),
            admin_key=admin_key,
            api_key_pepper=pepper,
            model_id=os.getenv("LAYA_MODEL_ID", "aac6fef/laya-mlx"),
            model_revision=revision,
            model_dtype=os.getenv("LAYA_MODEL_DTYPE", "float16"),
            eager_load_model=_bool_env("LAYA_EAGER_LOAD_MODEL", True),
            store_payloads=_bool_env("LAYA_STORE_PAYLOADS", True),
            trust_proxy_headers=_bool_env("LAYA_TRUST_PROXY_HEADERS", False),
            max_log_export_rows=int(os.getenv("LAYA_MAX_LOG_EXPORT_ROWS", "10000")),
            enabled_models=enabled_models,
            default_model=os.getenv("LAYA_DEFAULT_MODEL", "laya"),
            omni_kit_path=Path(omni_path).expanduser().resolve() if omni_path else None,
            max_upload_mb=float(os.getenv("LAYA_MAX_UPLOAD_MB", "50")),
        )
