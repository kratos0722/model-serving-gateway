import re

FEATURES = {"recency": 0.5, "ctr": 0.3, "price_norm": 0.8, "query_match": 1.0}

# A pragmatic parser for the Prometheus text exposition format: each sample
# line is `name{labels}? value`, with optional # HELP / # TYPE comment lines.
_SAMPLE = re.compile(
    r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+'
    r'(\+Inf|-Inf|[0-9eE.+\-]+)$'
)


def _parse_prometheus(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        assert _SAMPLE.match(line), f"invalid prometheus line: {line!r}"
        name, value = line.rsplit(" ", 1)
        samples[name.strip()] = float("inf") if value == "+Inf" else float(value)
    return samples


def test_metrics_parse_as_valid_prometheus(client):
    client.post("/predict", json={"features": FEATURES, "version": "v1"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    samples = _parse_prometheus(resp.text)
    assert samples  # non-empty
    # Required series present.
    assert any(k.startswith("inference_requests_total") for k in samples)
    assert any(k.startswith("inference_latency_seconds_bucket") for k in samples)
    assert "inference_latency_p99_seconds" in samples


def test_request_counter_increments(client):
    for _ in range(3):
        client.post("/predict", json={"features": FEATURES, "version": "v1"})
    samples = _parse_prometheus(client.get("/metrics").text)
    key = 'inference_requests_total{version="v1",outcome="positive"}'
    assert samples[key] == 3.0


def test_histogram_buckets_are_cumulative(client):
    for _ in range(5):
        client.post("/predict", json={"features": FEATURES, "version": "v1"})
    text = client.get("/metrics").text
    bucket_values = []
    for line in text.splitlines():
        if line.startswith('inference_latency_seconds_bucket{version="v1"'):
            value = float(line.rsplit(" ", 1)[1])
            bucket_values.append(value)
    assert bucket_values  # at least one bucket emitted
    # Cumulative histogram: non-decreasing, final (+Inf) equals total count.
    assert bucket_values == sorted(bucket_values)
    assert bucket_values[-1] == 5.0
