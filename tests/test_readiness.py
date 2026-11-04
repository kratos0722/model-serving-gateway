"""Tests for the /readyz readiness probe (distinct from /healthz liveness)."""


def test_readyz_ready_when_all_models_score(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    # Two artifacts ship with the repo (ranker v1 + v2).
    assert body["models_ready"] == len(client.app.state.registry.all())
    assert body["models_ready"] >= 2


def test_readyz_is_distinct_from_healthz(client):
    health = client.get("/healthz").json()
    ready = client.get("/readyz").json()
    # Liveness reports load count; readiness reports scoring-ready count.
    assert "models_loaded" in health
    assert "models_ready" in ready
    assert "models_loaded" not in ready


def test_readyz_503_when_a_model_cannot_score(client, monkeypatch):
    registry = client.app.state.registry
    healthy = registry.all()

    class BrokenModel:
        name = "ranker"
        version = "broken"
        feature_order = ("x",)

        def predict(self, _features):
            raise RuntimeError("corrupt artifact")

    monkeypatch.setattr(registry, "all", lambda: [*healthy, BrokenModel()])

    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert "ranker:broken" in body["unready_models"]
