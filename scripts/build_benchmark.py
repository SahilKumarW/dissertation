#!/usr/bin/env python
"""CLI entry point for Step 1: build and freeze the benchmark artefacts.

Usage:
    python scripts/build_benchmark.py
    python scripts/build_benchmark.py --source-dir /path/to/supplier/files
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.data.benchmark_builder import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
