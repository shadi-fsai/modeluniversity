import pytest
import shutil
import json
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for the test data directory."""
    return Path(__file__).parent / "data"


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

    # --- Teardown: remove it again after all tests complete
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
