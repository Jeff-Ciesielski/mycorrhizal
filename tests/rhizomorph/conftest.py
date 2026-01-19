"""Pytest configuration for rhizomorph tests"""

import pytest
from mycorrhizal.rhizomorph.core import _clear_interface_view_cache


@pytest.fixture(autouse=True)
def clear_interface_cache():
    """Clear the interface view cache before each test.

    This ensures that tests don't interfere with each other through
    cached interface views. The cache maps (blackboard_id, interface_type)
    to view instances, which can cause issues when tests reuse the same
    blackboard instance but expect different access patterns.
    """
    _clear_interface_view_cache()
    yield
    # Optional: clear after test as well
    _clear_interface_view_cache()
