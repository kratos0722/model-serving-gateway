"""Model artifacts and pure-function scoring.

A model is a small, versioned JSON artifact (logistic-regression weights, bias
and a stable feature order). Scoring is a pure function ``features -> prob``
implemented with NumPy so results are deterministic and dependency-light.

In a real system the artifact could just as well point at a serialized
PyTorch / ONNX / XGBoost blob; the registry + routing machinery around it does
not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


class FeatureError(ValueError):
    """Raised when a request is missing features the model requires."""


@dataclass(frozen=True)
class LogisticModel:
    """A versioned logistic-regression scorer loaded from a JSON artifact."""

    name: str
    version: str
    feature_order: tuple[str, ...]
    weights: np.ndarray
    bias: float
    description: str = ""
    _weights_list: tuple[float, ...] = field(default=(), repr=False)

    @classmethod
    def from_artifact(cls, data: Mapping[str, Any]) -> "LogisticModel":
        required = ("name", "version", "feature_order", "weights", "bias")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"artifact missing keys: {missing}")
        feature_order = tuple(str(f) for f in data["feature_order"])
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.shape != (len(feature_order),):
            raise ValueError(
                "weights length does not match feature_order "
                f"({weights.shape} vs {len(feature_order)})"
            )
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            feature_order=feature_order,
            weights=weights,
            bias=float(data["bias"]),
            description=str(data.get("description", "")),
            _weights_list=tuple(float(w) for w in weights.tolist()),
        )

    def vectorize(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise FeatureError(f"missing features: {missing}")
        try:
            return np.array(
                [float(features[f]) for f in self.feature_order], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:  # non-numeric feature value
            raise FeatureError("feature values must be numeric") from exc

    def predict_proba(self, features: Mapping[str, float]) -> float:
        x = self.vectorize(features)
        z = float(np.dot(self.weights, x) + self.bias)
        # Numerically stable logistic sigmoid.
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = float(np.exp(z))
        return float(ez / (1.0 + ez))

    def predict(self, features: Mapping[str, float]) -> tuple[float, int]:
        """Return ``(probability, label)`` where label is thresholded at 0.5."""
        prob = self.predict_proba(features)
        label = 1 if prob >= 0.5 else 0
        return prob, label

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "feature_order": list(self.feature_order),
            "weights": list(self._weights_list),
            "bias": self.bias,
            "description": self.description,
        }
