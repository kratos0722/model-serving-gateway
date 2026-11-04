import pytest
from fastapi.testclient import TestClient

from app.config import GatewayConfig
from app.main import create_app


@pytest.fixture
def config():
    """Default config: canary 90/10, shadow v2."""
    return GatewayConfig()


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as c:
        yield c


def make_client(config):
    app = create_app(config)
    return TestClient(app)
