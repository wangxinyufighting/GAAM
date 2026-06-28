"""Shared pytest configuration for GAAM training tests."""

import os
from pathlib import Path
import sys


# Torch/OpenMP can abort subprocess-based CLI tests on constrained macOS
# environments if these are not set before the first torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
