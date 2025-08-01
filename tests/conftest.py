import pytest
from app.config import Settings

@pytest.fixture(scope="session")
def settings():
    return Settings(polygon_api_key="demo")