#!/usr/bin/env python
"""Freeze the experiment (master spec sections 11 and 35).

    python scripts/freeze.py            # write the freeze manifest
    python scripts/freeze.py --verify   # check nothing has changed since

After freezing, no benchmark-driven tuning is permitted. Any change requires a
new experiment version, and --verify is what makes that checkable rather than
a matter of trust.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import repo_root  # noqa: E402
from georef.provenance.freeze import build_freeze_manifest, verify_against_freeze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify the experiment state.")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--experiment-version", default="v1")
    args = parser.parse_args()

    root = repo_root()
    path = root / "experiments" / "freeze_manifest.json"

    if args.verify:
        if not path.is_file():
            print("No freeze manifest. Run scripts/freeze.py first.", file=sys.stderr)
            return 2
        frozen = json.loads(path.read_text(encoding="utf-8"))
        differences = verify_against_freeze(root, frozen)
        if not differences:
            print(f"UNCHANGED since freeze {frozen['freeze_sha256'][:16]}")
            print(f"  frozen at  {frozen['frozen_at_utc']}")
            return 0
        print(f"CHANGED since freeze {frozen['freeze_sha256'][:16]}\n")
        for difference in differences:
            print(f"  {difference}")
        print(
            "\nA benchmark run against this state would not be the frozen experiment.\n"
            "Either revert the change or freeze a new experiment version."
        )
        return 1

    manifest = build_freeze_manifest(root, {"experiment_version": args.experiment_version})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Frozen: {manifest['freeze_sha256']}\n")
    print(f"  experiment version   {manifest['experiment_version']}")
    print(f"  data files hashed    {len(manifest['data_hashes'])}")
    print(f"  configs hashed       {len(manifest['config_hashes'])}")
    print(f"  source files hashed  {len(manifest['code_hashes'])}")
    print(f"  extraction prompt    {manifest['prompts']['extraction_version']}")
    print(f"  baseline prompt      {manifest['prompts']['baseline_version']}")
    print(f"  chroma records       {manifest['chroma'].get('record_count')}")
    print(f"\n  manifest   {path}")
    print("\n  From here, no benchmark-driven tuning. Verify with --verify before running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
