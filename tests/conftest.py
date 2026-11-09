from tenet_harness.testing import FakeModel, FakeProvider  # noqa: F401

import pytest


@pytest.fixture
def provider():
    return FakeProvider()
