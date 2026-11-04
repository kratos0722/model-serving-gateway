from app.config import GatewayConfig
from app.registry import ModelRegistry

from .conftest import make_client

FEATURES = {"recency": 0.5, "ctr": 0.3, "price_norm": 0.8, "query_match": 1.0}


def test_shadow_does_not_change_served_response():
    cfg = GatewayConfig().with_overrides(shadow_version="v2")
    reg = ModelRegistry.from_dir(cfg.models_dir)
    expected_primary = reg.get("ranker", "v1").predict_proba(FEATURES)

    with make_client(cfg) as client:
        resp = client.post(
            "/predict", json={"features": FEATURES, "version": "v1"}
        )
    body = resp.json()
    assert resp.status_code == 200
    assert body["version_served"] == "v1"
    assert body["shadowed"] is True
    assert body["shadow_version"] == "v2"
    # Served probability is the PRIMARY's, unaffected by the shadow.
    assert abs(body["probability"] - expected_primary) < 1e-12


def test_shadow_records_divergence():
    cfg = GatewayConfig().with_overrides(shadow_version="v2", shadow_threshold=0.05)
    with make_client(cfg) as client:
        client.post("/predict", json={"features": FEATURES, "version": "v1"})
        metrics_text = client.get("/metrics").text

    assert 'inference_shadow_requests_total{primary="v1",shadow="v2"} 1' in metrics_text
    # v1 vs v2 on these features diverge by > 0.05, so divergence is counted.
    assert (
        'inference_shadow_divergence_total{primary="v1",shadow="v2"} 1' in metrics_text
    )


def test_shadow_skipped_when_primary_equals_shadow():
    cfg = GatewayConfig().with_overrides(shadow_version="v1")
    with make_client(cfg) as client:
        resp = client.post(
            "/predict", json={"features": FEATURES, "version": "v1"}
        )
    assert resp.json()["shadowed"] is False
