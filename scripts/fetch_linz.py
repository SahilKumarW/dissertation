#!/usr/bin/env python
"""Step 2: fetch and freeze the LINZ NZ Place Names snapshot.

    python scripts/fetch_linz.py --describe     # verify key, layer, fields
    python scripts/fetch_linz.py                # full ingest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.linz.ingest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
