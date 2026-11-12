import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustmon import model  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    clf, (x_test, y_test) = model.train(seed=0)
    half = len(x_test) // 2
    return clf, x_test[:half], y_test[:half], x_test[half:], y_test[half:]
