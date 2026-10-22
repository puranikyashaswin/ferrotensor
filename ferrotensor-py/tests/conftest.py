import os

import pytest

import ferrotensor as ft

RUN_GPU = os.environ.get("FERROTENSOR_GPU_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if RUN_GPU:
        return
    skip = pytest.mark.skip(reason="set FERROTENSOR_GPU_TESTS=1 to run GPU tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU adapter (FERROTENSOR_GPU_TESTS=1)")


def close(a, b, tol=1e-5):
    """Compare nested lists/floats elementwise."""
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (a, b)
        for x, y in zip(a, b, strict=True):
            close(x, y, tol)
    else:
        assert a == pytest.approx(b, abs=tol, rel=tol), (a, b)


@pytest.fixture
def cpu():
    return ft.Device("cpu")
