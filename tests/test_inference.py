import math

from app.config import GatewayConfig
from app.registry import ModelRegistry


FEATURES = {"recency": 0.5, "ctr": 0.3, "price_norm": 0.8, "query_match": 1.0}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_v1_matches_hand_computed_value():
    reg = ModelRegistry.from_dir(GatewayConfig().models_dir)
    model = reg.get("ranker", "v1")

    # Hand-computed independently of the model code:
    # weights [1.2, 2.5, -0.8, 1.7], bias -0.5
    z = 1.2 * 0.5 + 2.5 * 0.3 - 0.8 * 0.8 + 1.7 * 1.0 - 0.5
    assert z == __import__("pytest").approx(1.91)
    expected = _sigmoid(z)

    prob, label = model.predict(FEATURES)
    assert prob == __import__("pytest").approx(expected, rel=1e-12)
    assert label == 1


def test_v2_differs_from_v1():
    reg = ModelRegistry.from_dir(GatewayConfig().models_dir)
    p1 = reg.get("ranker", "v1").predict_proba(FEATURES)
    p2 = reg.get("ranker", "v2").predict_proba(FEATURES)
    # Different weights must produce an observably different score.
    assert abs(p1 - p2) > 0.05


def test_predict_endpoint_returns_served_version(client):
    resp = client.post(
        "/predict", json={"features": FEATURES, "version": "v1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["version_served"] == "v1"
    assert body["label"] == 1
    assert 0.0 <= body["probability"] <= 1.0
