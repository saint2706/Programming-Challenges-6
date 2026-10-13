from __future__ import annotations

import pytest
from credit_scorecard import pipeline


@pytest.fixture(scope="session")
def result():
    """Real Give Me Some Credit data (from the parquet cache after the first fetch), trained once."""
    return pipeline.run()
