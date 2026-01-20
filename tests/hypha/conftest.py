"""Pytest configuration for hypha tests"""

import pytest
from mycorrhizal.common.compilation import _clear_compilation_cache


@pytest.fixture(autouse=True)
def clear_compilation_cache():
    """Clear the global compilation cache after each test.

    This ensures that tests don't interfere with each other through
    shared compilation metadata. The compilation cache stores interface
    type metadata that could become stale across test runs.

    Note: The interface view cache is instance-local (each runner has
    its own cache), so that doesn't need to be cleared. But the compilation
    cache is global and MUST be cleared between tests.
    """
    yield
    # Clear after test to prevent pollution
    _clear_compilation_cache()
