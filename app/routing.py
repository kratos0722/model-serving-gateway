"""Deterministic routing: canary split and A/B assignment.

Both strategies map a stable string key to a point in ``[0, 1)`` via a SHA-256
hash. Because the mapping is a pure function of the key (plus a config salt),
routing is fully reproducible and unit-testable without any randomness.
"""

from __future__ import annotations

import hashlib

from .config import GatewayConfig


def hash_unit(key: str, salt: str = "") -> float:
    """Map an arbitrary string key to a stable float in ``[0, 1)``."""
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16)  # 64 bits of entropy
    return bucket / float(1 << 64)


class Router:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        # Deterministic iteration order for the cumulative canary split.
        self._canary_items = sorted(config.canary_weights.items())

    def assign_canary(self, key: str) -> str:
        """Weighted traffic split. Returns the chosen version."""
        u = hash_unit(key, f"{self.config.hash_salt}:canary")
        cumulative = 0.0
        for version, weight in self._canary_items:
            cumulative += weight
            if u < cumulative:
                return version
        return self._canary_items[-1][0]  # float-rounding guard

    def assign_ab(self, user_id: str) -> tuple[str, str]:
        """Stable A/B assignment. Returns ``(variant, version)``."""
        variants = sorted(self.config.ab_variants.items())
        u = hash_unit(user_id, f"{self.config.hash_salt}:ab")
        idx = min(int(u * len(variants)), len(variants) - 1)
        variant, version = variants[idx]
        return variant, version

    def route(self, key: str, user_id: str | None) -> tuple[str, str | None]:
        """Pick the primary version per the configured strategy.

        Returns ``(version, variant)`` where ``variant`` is only set for A/B.
        """
        if self.config.routing_strategy == "ab":
            # A/B needs a stable identity; fall back to the request key.
            variant, version = self.assign_ab(user_id or key)
            return version, variant
        return self.assign_canary(key), None
