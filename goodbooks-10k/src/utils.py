from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def project_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def default_data_path(filename: str) -> Path:
    return project_path(filename)
