"""Application configuration via environment variables."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "LLM Eval Pipeline API"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    workspace_root: str = "."
    models_config_path: str = "config/models.yaml"
    tests_config_path: str = "config/tests.yaml"
    task_registry_path: str = "config/task_registry.yaml"
    reports_dir: str = "reports"
    generated_datasets_dir: str = "eval_datasets/generated"

    model_config = {
        "env_prefix": "EVAL_",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
