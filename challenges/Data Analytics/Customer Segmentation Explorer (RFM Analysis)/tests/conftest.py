import pytest
from helpers import population


@pytest.fixture(scope="session")
def pop():
    return population()
