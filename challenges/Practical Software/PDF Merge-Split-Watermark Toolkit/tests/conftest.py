import pytest
from helpers import make_pdf


@pytest.fixture
def three(tmp_path):
    return make_pdf(tmp_path / "three.pdf", 3)


@pytest.fixture
def five(tmp_path):
    return make_pdf(tmp_path / "five.pdf", 5)
