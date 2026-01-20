"""Pytest configuration for septum tests"""

import pytest
from mycorrhizal.septum.core import _clear_interface_view_cache
from mycorrhizal.common.compilation import _clear_compilation_cache


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear global caches after each test to ensure proper isolation.

    This fixture clears caches that can accumulate state mutations across tests:
    - Interface view cache: Type-specific interface views that may hold mutated state
    - Compilation cache: Interface type metadata that can become stale

    Note: We do NOT clear the state registry because states are registered at
    module import time. Clearing the registry would require re-importing modules,
    which is not feasible in pytest's execution model.
    """
    yield
    # Clear after test to prevent pollution of subsequent tests
    _clear_interface_view_cache()
    _clear_compilation_cache()
