"""Model registry: load and cache versioned artifacts at startup."""

from __future__ import annotations

import json
from pathlib import Path

from .model import LogisticModel


class UnknownModelError(KeyError):
    """Raised when a requested model name/version is not registered."""


class ModelRegistry:
    """Loads all ``<models_dir>/<name>/<version>.json`` artifacts and caches them."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], LogisticModel] = {}

    @classmethod
    def from_dir(cls, models_dir: Path) -> "ModelRegistry":
        reg = cls()
        models_dir = Path(models_dir)
        if not models_dir.is_dir():
            raise FileNotFoundError(f"models dir not found: {models_dir}")
        for artifact in sorted(models_dir.glob("*/*.json")):
            with artifact.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            model = LogisticModel.from_artifact(data)
            reg.register(model)
        if not reg._models:
            raise FileNotFoundError(f"no model artifacts found under {models_dir}")
        return reg

    def register(self, model: LogisticModel) -> None:
        self._models[(model.name, model.version)] = model

    def get(self, name: str, version: str) -> LogisticModel:
        try:
            return self._models[(name, version)]
        except KeyError as exc:
            raise UnknownModelError(f"{name}:{version}") from exc

    def versions(self, name: str) -> list[str]:
        return sorted(v for (n, v) in self._models if n == name)

    def list_models(self) -> list[dict]:
        return [self._models[key].info() for key in sorted(self._models)]

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._models
