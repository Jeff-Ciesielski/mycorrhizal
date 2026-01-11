#!/usr/bin/env python3
"""
Spores Encoders

Encoders for serializing OCEL LogRecords to various formats.
"""

from .base import Encoder
from .json import JSONEncoder

__all__ = ['Encoder', 'JSONEncoder']
