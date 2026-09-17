"""Shared pytest fixtures.

Settings requires POLYGON_API_KEY, so a dummy value is injected before anything
imports app.config. Tests that need a live key check `has_live_polygon_key`.
"""
import os

import pytest

DUMMY_POLYGON_KEY = "test-dummy-key"
os.environ.setdefault("POLYGON_API_KEY", DUMMY_POLYGON_KEY)

from app.config import Settings  # noqa: E402  (must run after the env default)


def has_live_polygon_key() -> bool:
    return os.environ.get("POLYGON_API_KEY", DUMMY_POLYGON_KEY) != DUMMY_POLYGON_KEY


requires_live_polygon = pytest.mark.skipif(
    not has_live_polygon_key(),
    reason="needs a real POLYGON_API_KEY (network access)",
)


@pytest.fixture(scope="session")
def settings():
    return Settings(polygon_api_key="demo")
