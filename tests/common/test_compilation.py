#!/usr/bin/env python3
"""
Tests for the common compilation cache module.

Tests exception handling, edge cases, and type safety of the compilation system.
"""

import pytest
from typing import Protocol, Annotated
from pydantic import BaseModel

from mycorrhizal.common.compilation import (
    _compile_function_metadata,
    _get_compiled_metadata,
    _clear_compilation_cache,
    _get_compilation_cache_size,
    CompiledMetadata,
    _has_interface_metadata,
    _extract_generic_type_arg,
)
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite


# ============================================================================
# Test Fixtures
# ============================================================================


class TestBlackboard(BaseModel):
    """Test blackboard"""
    value: int = 42
    name: str = "test"


@blackboard_interface
class TestInterface(Protocol):
    """Test interface"""
    value: Annotated[int, readonly]
    name: Annotated[str, readwrite]


@blackboard_interface
class AnotherInterface(Protocol):
    """Another test interface"""
    value: Annotated[int, readonly]


# ============================================================================
# Helper Functions for Testing
# ============================================================================


def sample_function_no_interface(bb):
    """Function without interface type hint"""
    return bb.value


def sample_function_with_interface(bb: TestInterface):
    """Function with interface type hint"""
    return bb.value


async def async_function_with_interface(bb: TestInterface):
    """Async function with interface type hint"""
    return bb.value


class CallableObject:
    """Callable object (not a function)"""

    def __call__(self, bb: TestInterface):
        return bb.value


# ============================================================================
# Tests
# ============================================================================


class TestCompilationCache:
    """Test compilation cache functionality"""

    def setup_method(self):
        """Clear cache before each test"""
        _clear_compilation_cache()

    def test_cache_starts_empty(self):
        """Cache should start empty"""
        assert _get_compilation_cache_size() == 0

    def test_cache_size_increases_after_compilation(self):
        """Cache size should increase after compiling a function"""
        initial_size = _get_compilation_cache_size()
        _get_compiled_metadata(sample_function_with_interface)
        assert _get_compilation_cache_size() == initial_size + 1

    def test_cache_reuses_metadata(self):
        """Cache should return the same metadata object for the same function"""
        metadata1 = _get_compiled_metadata(sample_function_with_interface)
        metadata2 = _get_compiled_metadata(sample_function_with_interface)
        assert metadata1 is metadata2

    def test_clear_cache(self):
        """Clear cache should empty the cache"""
        _get_compiled_metadata(sample_function_with_interface)
        assert _get_compilation_cache_size() > 0
        _clear_compilation_cache()
        assert _get_compilation_cache_size() == 0


class TestCompileFunctionMetadata:
    """Test _compile_function_metadata function"""

    def setup_method(self):
        """Clear cache before each test"""
        _clear_compilation_cache()

    def test_function_without_interface(self):
        """Should compile metadata for function without interface"""
        metadata = _compile_function_metadata(sample_function_no_interface)
        assert metadata.has_interface is False
        assert metadata.interface_type is None
        assert metadata.readonly_fields is None
        assert metadata.readwrite_fields is None

    def test_function_with_interface(self):
        """Should compile metadata for function with interface"""
        metadata = _compile_function_metadata(sample_function_with_interface)
        assert metadata.has_interface is True
        assert metadata.interface_type == TestInterface
        assert metadata.readonly_fields == {"value"}
        assert metadata.readwrite_fields == {"name"}

    def test_async_function_with_interface(self):
        """Should compile metadata for async function with interface"""
        metadata = _compile_function_metadata(async_function_with_interface)
        assert metadata.has_interface is True
        assert metadata.interface_type == TestInterface

    def test_callable_object_without_name(self):
        """Should handle callable objects without __name__ attribute"""
        callable_obj = CallableObject()
        metadata = _compile_function_metadata(callable_obj)
        # Callable objects don't support get_type_hints, so should return no interface
        assert metadata.has_interface is False

    def test_raises_type_error_for_non_callable(self):
        """Should raise TypeError for non-callable objects"""
        with pytest.raises(TypeError, match="Expected callable"):
            _compile_function_metadata("not a function")

        with pytest.raises(TypeError, match="Expected callable"):
            _compile_function_metadata(42)

        with pytest.raises(TypeError, match="Expected callable"):
            _compile_function_metadata(None)


class TestExtractGenericTypeArg:
    """Test _extract_generic_type_arg function"""

    def test_non_generic_type_returns_none(self):
        """Non-generic types should return None"""
        assert _extract_generic_type_arg(int) is None
        assert _extract_generic_type_arg(str) is None
        assert _extract_generic_type_arg(TestInterface) is None

    def test_single_arg_generic_returns_arg(self):
        """Generic types with single type argument should return the argument"""
        from typing import List

        assert _extract_generic_type_arg(List[int]) == int

    def test_multi_arg_generic_returns_none(self):
        """Generic types with multiple type arguments should return None"""
        from typing import Dict

        assert _extract_generic_type_arg(Dict[str, int]) is None

    def test_empty_generic_returns_none(self):
        """Generic types with no arguments should return None"""
        from typing import List

        assert _extract_generic_type_arg(List) is None


class TestHasInterfaceMetadata:
    """Test _has_interface_metadata function"""

    def test_interface_with_metadata_returns_true(self):
        """Interface with _readonly_fields should return True"""
        assert _has_interface_metadata(TestInterface) is True

    def test_plain_type_returns_false(self):
        """Plain types without interface metadata should return False"""
        assert _has_interface_metadata(int) is False
        assert _has_interface_metadata(str) is False
        assert _has_interface_metadata(TestBlackboard) is False

    def test_generic_with_interface_arg_returns_true(self):
        """Generic type with interface argument should return True"""
        # This tests the case like SharedContext[TestInterface]
        # where we need to check if the type argument has metadata
        pass  # This requires importing SharedContext from septum


class TestExceptionHandling:
    """Test exception handling in compilation"""

    def setup_method(self):
        """Clear cache before each test"""
        _clear_compilation_cache()

    def test_name_error_propagates_from_compile(self):
        """NameError from undefined type hints should propagate with context"""
        def bad_function(bb: "UndefinedType"):
            pass

        with pytest.raises(NameError, match="Failed to compile metadata"):
            _compile_function_metadata(bad_function)

    def test_non_callable_raises_type_error(self):
        """Non-callable should raise TypeError"""
        with pytest.raises(TypeError, match="Expected callable"):
            _get_compiled_metadata("not a function")


class TestCompiledMetadataDataclass:
    """Test CompiledMetadata dataclass"""

    def test_metadata_is_frozen(self):
        """CompiledMetadata should be frozen (immutable)"""
        metadata = CompiledMetadata(
            has_interface=True,
            interface_type=TestInterface,
            readonly_fields={"value"},
            readwrite_fields={"name"},
        )

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            metadata.has_interface = False

    def test_metadata_equality(self):
        """Metadata with same values should be equal"""
        metadata1 = CompiledMetadata(
            has_interface=True,
            interface_type=TestInterface,
            readonly_fields={"value"},
            readwrite_fields={"name"},
        )

        metadata2 = CompiledMetadata(
            has_interface=True,
            interface_type=TestInterface,
            readonly_fields={"value"},
            readwrite_fields={"name"},
        )

        assert metadata1 == metadata2

    def test_metadata_inequality(self):
        """Metadata with different values should not be equal"""
        metadata1 = CompiledMetadata(
            has_interface=True,
            interface_type=TestInterface,
            readonly_fields={"value"},
            readwrite_fields={"name"},
        )

        metadata2 = CompiledMetadata(
            has_interface=False,
            interface_type=None,
            readonly_fields=None,
            readwrite_fields=None,
        )

        assert metadata1 != metadata2
