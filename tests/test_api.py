from app.config import GatewayConfig

from .conftest import make_client

FEATURES = {"recency": 0.5, "ctr": 0.3, "price_norm": 0.8, "query_match": 1.0}


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 2


def test_models_listing(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    versions = {m["version"] for m in resp.json()}
    assert versions == {"v1", "v2"}


def test_validation_error_returns_422(client):
    # Non-numeric feature value.
    resp = client.post(
        "/predict", json={"features": {"recency": "not-a-number"}}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_extra_fields_rejected(client):
    resp = client.post(
        "/predict",
        json={"features": FEATURES, "version": "v1", "bogus": 1},
    )
    assert resp.status_code == 422


def test_missing_features_returns_422(client):
    # Passes schema (dict[str, float]) but omits required model features.
    resp = client.post(
        "/predict", json={"features": {"recency": 0.5}, "version": "v1"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_features"


def test_unknown_version_is_sanitized_404(client):
    resp = client.post(
        "/predict", json={"features": FEATURES, "version": "v999"}
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "unknown_model"
    # No internal identifiers / stack traces leaked.
    assert "v999" not in body["detail"]
    assert "Traceback" not in body["detail"]


def test_ab_endpoint_same_user_same_version():
    cfg = GatewayConfig().with_overrides(
        routing_strategy="ab", shadow_version=""
    )
    with make_client(cfg) as client:
        first = client.post(
            "/predict", json={"features": FEATURES, "user_id": "alice"}
        ).json()
        for _ in range(10):
            again = client.post(
                "/predict", json={"features": FEATURES, "user_id": "alice"}
            ).json()
            assert again["version_served"] == first["version_served"]
            assert again["variant"] == first["variant"]
        assert first["variant"] in ("A", "B")
