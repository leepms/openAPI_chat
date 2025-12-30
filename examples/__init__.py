"""Examples for OpenAI Chat API Client

This directory contains example scripts demonstrating various features:

- example.py: Basic usage examples
- examples_complete.py: Comprehensive feature demonstrations
- examples_v0.3.py: New features in version 0.3.0
- manual_test.py: Manual testing script for API endpoints

Usage:
    python examples/example.py
    python examples/examples_v0.3.py
"""

# Make examples importable
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
