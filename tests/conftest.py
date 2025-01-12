import pytest
import shutil
import json
from pathlib import Path
from datetime import datetime


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture for the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def mock_data_dir(test_data_dir) -> Path:
    """Fixture for the test data directory."""
    return test_data_dir / "mock_data"


@pytest.fixture(scope="session", autouse=True)
def test_outputs_dir(test_data_dir):
    """
    This fixture runs once before any tests start (setup)
    and once after all tests finish (teardown).
    """
    outputs_dir = test_data_dir / "outputs"

    # --- Setup: remove and recreate outputs_dir before any test runs
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir()

    # Yield control back to the test session
    yield outputs_dir

    # --- Teardown: rename it with a timestamp after all tests complete
    if outputs_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        outputs_dir.rename(test_data_dir / f"outputs_{timestamp}")
