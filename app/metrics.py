"""Hand-rolled Prometheus-style metrics (no prometheus_client dependency).

Exposes, in the Prometheus text exposition format:
  * inference_requests_total{version,outcome}          counter
  * inference_latency_seconds{version} histogram       (_bucket/_sum/_count)
  * inference_latency_p99_seconds                      gauge (rolling window)
  * inference_shadow_requests_total{primary,shadow}    counter
  * inference_shadow_divergence_total{primary,shadow}  counter
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import numpy as np

# Histogram bucket upper bounds in seconds (Prometheus "le" semantics).
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)


def _fmt(value: float) -> str:
    """Format a float the way the Prometheus text format expects."""
    if value == float("inf"):
        return "+Inf"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class Metrics:
    def __init__(
        self,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        window_size: int = 1024,
    ) -> None:
        self._buckets = tuple(buckets)
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str], int] = defaultdict(int)
        # version -> {bucket_upper_bound: count}, plus sum/count
        self._hist_counts: dict[str, list[int]] = defaultdict(
            lambda: [0] * (len(self._buckets) + 1)
        )
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_count: dict[str, int] = defaultdict(int)
        self._shadow_requests: dict[tuple[str, str], int] = defaultdict(int)
        self._shadow_divergence: dict[tuple[str, str], int] = defaultdict(int)
        self._latencies: deque[float] = deque(maxlen=window_size)

    # ---- recording -------------------------------------------------------

    def record_request(self, version: str, outcome: str) -> None:
        with self._lock:
            self._requests[(version, outcome)] += 1

    def observe_latency(self, version: str, seconds: float) -> None:
        with self._lock:
            counts = self._hist_counts[version]
            placed = False
            for i, upper in enumerate(self._buckets):
                if seconds <= upper:
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1  # the +Inf bucket
            self._hist_sum[version] += seconds
            self._hist_count[version] += 1
            self._latencies.append(seconds)

    def record_shadow(self, primary: str, shadow: str, diverged: bool) -> None:
        with self._lock:
            self._shadow_requests[(primary, shadow)] += 1
            if diverged:
                self._shadow_divergence[(primary, shadow)] += 1

    # ---- reading ---------------------------------------------------------

    def p99_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return float(np.percentile(np.array(self._latencies), 99))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": dict(self._requests),
                "shadow_requests": dict(self._shadow_requests),
                "shadow_divergence": dict(self._shadow_divergence),
                "hist_count": dict(self._hist_count),
            }

    def render(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP inference_requests_total Total inference requests.")
            lines.append("# TYPE inference_requests_total counter")
            for (version, outcome), value in sorted(self._requests.items()):
                lines.append(
                    f'inference_requests_total{{version="{version}",'
                    f'outcome="{outcome}"}} {value}'
                )

            lines.append(
                "# HELP inference_latency_seconds Inference latency in seconds."
            )
            lines.append("# TYPE inference_latency_seconds histogram")
            for version in sorted(self._hist_counts):
                counts = self._hist_counts[version]
                cumulative = 0
                for i, upper in enumerate(self._buckets):
                    cumulative += counts[i]
                    lines.append(
                        f'inference_latency_seconds_bucket{{version="{version}",'
                        f'le="{_fmt(upper)}"}} {cumulative}'
                    )
                cumulative += counts[-1]
                lines.append(
                    f'inference_latency_seconds_bucket{{version="{version}",'
                    f'le="+Inf"}} {cumulative}'
                )
                lines.append(
                    f'inference_latency_seconds_sum{{version="{version}"}} '
                    f"{_fmt(self._hist_sum[version])}"
                )
                lines.append(
                    f'inference_latency_seconds_count{{version="{version}"}} '
                    f"{self._hist_count[version]}"
                )

            p99 = (
                float(np.percentile(np.array(self._latencies), 99))
                if self._latencies
                else 0.0
            )
            lines.append(
                "# HELP inference_latency_p99_seconds p99 latency over a bounded "
                "rolling window."
            )
            lines.append("# TYPE inference_latency_p99_seconds gauge")
            lines.append(f"inference_latency_p99_seconds {_fmt(p99)}")

            lines.append(
                "# HELP inference_shadow_requests_total Requests scored by a "
                "shadow model."
            )
            lines.append("# TYPE inference_shadow_requests_total counter")
            for (primary, shadow), value in sorted(self._shadow_requests.items()):
                lines.append(
                    f'inference_shadow_requests_total{{primary="{primary}",'
                    f'shadow="{shadow}"}} {value}'
                )

            lines.append(
                "# HELP inference_shadow_divergence_total Shadow predictions that "
                "diverged from primary."
            )
            lines.append("# TYPE inference_shadow_divergence_total counter")
            for (primary, shadow), value in sorted(self._shadow_divergence.items()):
                lines.append(
                    f'inference_shadow_divergence_total{{primary="{primary}",'
                    f'shadow="{shadow}"}} {value}'
                )

            return "\n".join(lines) + "\n"
