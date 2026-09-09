#!/usr/bin/env python
"""Verify a frozen LINZ snapshot: contents plus diagnostic anchor coverage.

    python scripts/verify_linz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.linz.verify import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
