from app.config import GatewayConfig
from app.routing import Router, hash_unit


def test_hash_unit_is_deterministic_and_bounded():
    for key in ["a", "user-1", "xyz", ""]:
        v1 = hash_unit(key, "salt")
        v2 = hash_unit(key, "salt")
        assert v1 == v2
        assert 0.0 <= v1 < 1.0
    # Different salts separate the streams.
    assert hash_unit("k", "s1") != hash_unit("k", "s2")


def test_ab_assignment_is_stable_per_user():
    router = Router(GatewayConfig())
    for user in ["alice", "bob", "carol", "user-42", "9999"]:
        first = router.assign_ab(user)
        for _ in range(20):
            assert router.assign_ab(user) == first
        variant, version = first
        assert variant in ("A", "B")
        assert version in ("v1", "v2")


def test_ab_assignment_roughly_balanced():
    router = Router(GatewayConfig())
    counts = {"A": 0, "B": 0}
    n = 20000
    for i in range(n):
        variant, _ = router.assign_ab(f"user-{i}")
        counts[variant] += 1
    # ~50/50 split with generous tolerance.
    assert abs(counts["A"] / n - 0.5) < 0.03


def test_canary_split_matches_weights():
    cfg = GatewayConfig().with_overrides(canary_weights={"v1": 0.9, "v2": 0.1})
    router = Router(cfg)
    counts = {"v1": 0, "v2": 0}
    n = 20000
    for i in range(n):
        version = router.assign_canary(f"key-{i}")
        counts[version] += 1
    v2_fraction = counts["v2"] / n
    assert abs(v2_fraction - 0.1) < 0.02


def test_canary_is_deterministic():
    router = Router(GatewayConfig())
    keys = [f"key-{i}" for i in range(100)]
    first = [router.assign_canary(k) for k in keys]
    second = [router.assign_canary(k) for k in keys]
    assert first == second


def test_route_respects_ab_strategy():
    cfg = GatewayConfig().with_overrides(routing_strategy="ab")
    router = Router(cfg)
    version, variant = router.route("ignored", "alice")
    assert variant in ("A", "B")
    assert version in ("v1", "v2")
