"""
utils.py
========
Helper utilities for the ML Experimental Design project.
"""

import time
from contextlib import contextmanager

@contextmanager
def runtime_tracker(name: str):
    """Context manager to track the runtime of a block of code."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[Runtime] {name}: {elapsed:.4f} seconds")
