"""Gateway configuration.

Routing / canary / shadow behaviour is driven by a single immutable config
object with sane defaults. Values can be overridden via environment variables
so the running behaviour is easy to see and change without code edits.

Environment variables (all optional):
    MSG_MODEL_NAME          model name to serve                 (default: ranker)
    MSG_MODELS_DIR          directory holding model artifacts   (default: ./models)
    MSG_ROUTING_STRATEGY    "canary" | "ab"                     (default: canary)
    MSG_CANARY_WEIGHTS      JSON, e.g. {"v1":0.9,"v2":0.1}      (default: 90/10)
    MSG_AB_VARIANTS         JSON, e.g. {"A":"v1","B":"v2"}      (default: A->v1,B->v2)
    MSG_SHADOW_VERSION      version to shadow, "" to disable    (default: v2)
    MSG_SHADOW_THRESHOLD    prob divergence threshold           (default: 0.05)
    MSG_REQUEST_TIMEOUT_S   per-request scorer timeout seconds  (default: 1.0)
    MSG_LATENCY_WINDOW      rolling window size for p99         (default: 1024)
    MSG_HASH_SALT           salt for deterministic hashing      (default: msg)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class GatewayConfig:
    model_name: str = "ranker"
    models_dir: Path = DEFAULT_MODELS_DIR
    routing_strategy: str = "canary"  # "canary" | "ab"
    canary_weights: Mapping[str, float] = field(
        default_factory=lambda: {"v1": 0.9, "v2": 0.1}
    )
    ab_variants: Mapping[str, str] = field(
        default_factory=lambda: {"A": "v1", "B": "v2"}
    )
    shadow_version: str = "v2"
    shadow_threshold: float = 0.05
    request_timeout_s: float = 1.0
    latency_window: int = 1024
    hash_salt: str = "msg"

    def __post_init__(self) -> None:
        # Normalise canary weights to sum to 1.0 for a well-formed split.
        total = sum(self.canary_weights.values())
        if total <= 0:
            raise ValueError("canary_weights must sum to a positive value")
        normalised = {k: v / total for k, v in self.canary_weights.items()}
        object.__setattr__(self, "canary_weights", normalised)
        if self.routing_strategy not in ("canary", "ab"):
            raise ValueError(f"unknown routing_strategy: {self.routing_strategy}")

    def with_overrides(self, **kwargs) -> "GatewayConfig":
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GatewayConfig":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if "MSG_MODEL_NAME" in env:
            kwargs["model_name"] = env["MSG_MODEL_NAME"]
        if "MSG_MODELS_DIR" in env:
            kwargs["models_dir"] = Path(env["MSG_MODELS_DIR"])
        if "MSG_ROUTING_STRATEGY" in env:
            kwargs["routing_strategy"] = env["MSG_ROUTING_STRATEGY"]
        if "MSG_CANARY_WEIGHTS" in env:
            kwargs["canary_weights"] = {
                str(k): float(v)
                for k, v in json.loads(env["MSG_CANARY_WEIGHTS"]).items()
            }
        if "MSG_AB_VARIANTS" in env:
            kwargs["ab_variants"] = {
                str(k): str(v) for k, v in json.loads(env["MSG_AB_VARIANTS"]).items()
            }
        if "MSG_SHADOW_VERSION" in env:
            kwargs["shadow_version"] = env["MSG_SHADOW_VERSION"]
        if "MSG_SHADOW_THRESHOLD" in env:
            kwargs["shadow_threshold"] = float(env["MSG_SHADOW_THRESHOLD"])
        if "MSG_REQUEST_TIMEOUT_S" in env:
            kwargs["request_timeout_s"] = float(env["MSG_REQUEST_TIMEOUT_S"])
        if "MSG_LATENCY_WINDOW" in env:
            kwargs["latency_window"] = int(env["MSG_LATENCY_WINDOW"])
        if "MSG_HASH_SALT" in env:
            kwargs["hash_salt"] = env["MSG_HASH_SALT"]
        return cls(**kwargs)
