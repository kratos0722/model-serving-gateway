# Model Serving Gateway

A production-style **FastAPI online-inference gateway** that serves versioned
model artifacts with **canary rollout, shadow traffic, deterministic A/B
assignment, and Prometheus-style metrics**. It owns the flow from a model
artifact on disk to an online prediction over HTTP.

The scorers here are deliberately light and deterministic — logistic-regression
weights evaluated with NumPy — so the repo stays dependency-free and every test
is reproducible. The routing, shadowing, metrics and robustness machinery around
them is the real subject, and it generalizes directly to heavyweight backends
(see [Generalizing to real model servers](#generalizing-to-real-model-servers)).

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
  POST /predict          │                FastAPI app                    │
  {features, user_id,    │                                               │
   version?}             │   1. validate (pydantic v2)  ── 422 on bad ───┼──▶ sanitized error
        │                │                                               │
        ▼                │   2. route                                    │
  ┌───────────┐          │      ├─ canary: weighted split by hash(key)   │
  │  Router   │──────────┼──────┤                                        │
  └───────────┘          │      └─ A/B: stable variant by hash(user_id)  │
                         │                                               │
                         │   3. ModelRegistry.get(name, version) ─ 404 ──┼──▶ sanitized error
                         │          │  (cached at startup)               │
                         │          ▼                                    │
                         │   4. score primary  ── timeout ───────────────┼──▶ 503
                         │          │   (NumPy, timeout-guarded)         │
                         │          ▼                                    │
                         │   5. shadow version scored in-line;           │
                         │      divergence recorded; result DISCARDED    │
                         │          │                                    │
                         │          ▼                                    │
                         │   6. Metrics.record(...)                      │
                         │          │                                    │
                         └──────────┼────────────────────────────────────┘
                                    ▼
                         200 {version_served, variant, probability,
                              label, shadowed, shadow_version}

  GET /healthz   liveness + models loaded
  GET /readyz    readiness: 200 only if every model can score, else 503
  GET /models    registry listing (name, version, weights, feature order)
  GET /metrics   Prometheus text exposition
```

`/healthz` is a **liveness** probe (the process is up). `/readyz` is a
**readiness** probe: it scores every registered model with a zero vector and
returns `200` only when all succeed, otherwise `503` with the unready versions —
so a pod that loaded a corrupt artifact is kept out of rotation.

### Components

| Module             | Responsibility                                                      |
|--------------------|---------------------------------------------------------------------|
| `app/model.py`     | `LogisticModel`: pure NumPy `features -> prob`, loaded from JSON     |
| `app/registry.py`  | `ModelRegistry`: loads/caches `models/<name>/<version>.json`         |
| `app/routing.py`   | `Router`: deterministic canary split + A/B assignment via SHA-256   |
| `app/metrics.py`   | `Metrics`: hand-rolled counters, histogram, rolling-window p99       |
| `app/config.py`    | `GatewayConfig`: immutable config with env overrides + sane defaults |
| `app/schemas.py`   | pydantic v2 request/response models                                 |
| `app/main.py`      | FastAPI wiring, routes, sanitized error handlers, timeout guard      |

---

## Model artifacts

A model version is a small JSON artifact under `models/`:

```jsonc
// models/ranker/v1.json
{
  "name": "ranker",
  "version": "v1",
  "feature_order": ["recency", "ctr", "price_norm", "query_match"],
  "weights": [1.2, 2.5, -0.8, 1.7],
  "bias": -0.5
}
```

The score is a logistic regression: `prob = sigmoid(weights · features + bias)`.
`v1` and `v2` ship with **different weights** so routing decisions are
observable end-to-end. The `ModelRegistry` loads and caches every artifact at
startup; scoring is a pure function with no I/O.

---

## Routing semantics

All routing is a **pure function of a stable key** (SHA-256 → a float in
`[0, 1)`, salted from config), so it is fully reproducible and unit-testable
with no randomness.

- **Canary (default).** Weighted traffic split across versions, e.g.
  `v1=0.9, v2=0.1`. Weights are normalized and applied as a cumulative split
  over `hash(routing_key)`. Over many keys the realized split matches the
  configured weights (tested to within tolerance).
- **A/B.** Deterministic variant assignment by stable hash of `user_id`. The
  same user always lands in the same variant (`A → v1`, `B → v2` by default),
  which is the property an experiment needs to attribute outcomes.
- **Shadow.** A configured shadow version (default `v2`) is scored **in-line on
  every request**, but its output never affects the served response. The
  gateway records a `shadow_requests_total` counter and a
  `shadow_divergence_total` counter (incremented when the shadow's probability
  differs from the primary's by more than `shadow_threshold`). A failing shadow
  (timeout / bad features) is counted as a divergence and still cannot break the
  primary response.

The served response always comes from the **primary** version; a dedicated test
asserts the served probability equals the primary model's output even while a
shadow is running.

---

## Metrics

`GET /metrics` returns the Prometheus **text exposition format** (version
`0.0.4`), hand-rolled — no `prometheus_client` dependency required.

| Metric                                                  | Type      | Labels            |
|---------------------------------------------------------|-----------|-------------------|
| `inference_requests_total`                              | counter   | `version,outcome` |
| `inference_latency_seconds` (`_bucket` / `_sum` / `_count`) | histogram | `version` + `le`  |
| `inference_latency_p99_seconds`                         | gauge     | —                 |
| `inference_shadow_requests_total`                       | counter   | `primary,shadow`  |
| `inference_shadow_divergence_total`                     | counter   | `primary,shadow`  |

The histogram uses fixed buckets (`0.001 … 1.0`, plus `+Inf`); the p99 gauge is
computed from a **bounded rolling window** of recent latencies (`latency_window`,
default 1024) so it stays memory-stable under sustained traffic.

---

## Robustness

- **Input validation → 422** via pydantic v2; unknown extra fields are rejected.
- **Missing model features → 422** with a sanitized body.
- **Unknown model/version → 404** with a generic message (no internal
  identifiers leaked).
- **Per-request timeout** guards the scorer (`request_timeout_s`, default 1.0s);
  exceeding it returns `503` instead of hanging.
- **No stack traces ever reach clients** — a catch-all handler returns a
  sanitized `500` body.

---

## Configuration

Driven by an immutable `GatewayConfig` with env overrides (all optional):

| Env var                 | Default              | Meaning                              |
|-------------------------|----------------------|--------------------------------------|
| `MSG_MODEL_NAME`        | `ranker`             | model name to serve                  |
| `MSG_MODELS_DIR`        | `./models`           | artifact directory                   |
| `MSG_ROUTING_STRATEGY`  | `canary`             | `canary` or `ab`                     |
| `MSG_CANARY_WEIGHTS`    | `{"v1":0.9,"v2":0.1}`| JSON weighted split                  |
| `MSG_AB_VARIANTS`       | `{"A":"v1","B":"v2"}`| JSON variant → version map           |
| `MSG_SHADOW_VERSION`    | `v2`                 | version to shadow (`""` disables)    |
| `MSG_SHADOW_THRESHOLD`  | `0.05`               | prob divergence threshold            |
| `MSG_REQUEST_TIMEOUT_S` | `1.0`                | per-request scorer timeout (seconds) |
| `MSG_LATENCY_WINDOW`    | `1024`               | rolling window size for p99          |
| `MSG_HASH_SALT`         | `msg`                | salt for deterministic hashing       |

---

## Run

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

### Example requests

```bash
# Predict (canary-routed; user_id gives a stable routing key)
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"recency":0.5,"ctr":0.3,"price_norm":0.8,"query_match":1.0},"user_id":"user-123"}'
# -> {"model_name":"ranker","version_served":"v1","variant":null,
#     "probability":0.871...,"label":1,"shadowed":true,"shadow_version":"v2"}

# Force a specific version
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"recency":0.5,"ctr":0.3,"price_norm":0.8,"query_match":1.0},"version":"v2"}'

# Metrics (Prometheus text format)
curl -s http://localhost:8000/metrics

# Registry listing
curl -s http://localhost:8000/models
```

---

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage includes: inference correctness vs. a hand-computed value, deterministic
A/B (same `user_id` → same variant), canary split matching configured weights
over many keys, shadow not changing the served response but recording divergence,
`/metrics` incrementing and parsing as valid Prometheus text, `422` on invalid
input, and a sanitized `404` on unknown versions.

---

## Generalizing to real model servers

This repo keeps the scorer trivial on purpose; the surrounding pattern is what
transfers:

- **Artifacts.** Swap the JSON logistic weights for a real artifact pointer — a
  `state_dict`, an ONNX/TorchScript blob, an XGBoost booster — resolved from a
  model store (S3/GCS, MLflow, a model registry). `ModelRegistry` becomes the
  thing that pins versions and warms caches.
- **Scoring backends.** `LogisticModel.predict` stands in for a call into
  **Ray Serve**, **NVIDIA Triton**, or **TorchServe**. The timeout guard,
  canary/A-B routing, shadow comparison and metrics layer sit *in front* of any
  of those unchanged.
- **Features & tokenizers.** `features -> vector` here is a dict lookup; in
  production it fans out to a **feature store** (Feast/Tecton) for online
  features, or a **tokenizer** + preprocessing graph for an LLM / vision model.
- **Metrics.** The hand-rolled exposition would be replaced by `prometheus_client`
  (or OpenTelemetry) scraped by Prometheus/Grafana, with the same series names.
- **Shadow & canary.** The exact divergence-tracking + weighted-rollout loop is
  how you de-risk shipping a new model: shadow it against production traffic,
  watch divergence, then ramp the canary weight.
