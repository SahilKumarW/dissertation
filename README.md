# dissertation-georef

MSc Artificial Intelligence dissertation: natural-language spatial
georeferencing. A controlled A/B comparison between a standalone small-LLM
baseline (Condition A) and a retrieval-grounded, LINZ-grounded, deterministic-GIS
hybrid pipeline (Condition B), evaluated on WGS84 geodesic positional error.

**Current state: Step 1 complete.** The benchmark, ground truth and dev pool are
built, hashed and tested. No prediction code exists yet.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Step 1 — build the benchmark

Copy the two supplier files into `data/source/`:

```
data/source/SampleDataNoCoords.xlsx
data/source/MWLRDataCHR20230307_nz_only_with_coordinates_desc_60_or_more.csv
```

Then:

```bash
python scripts/build_benchmark.py
python -m pytest -q
```

The build is deterministic: the same inputs and the same
`configs/benchmark.yaml` reproduce identical file hashes.

Read `docs/BENCHMARK_CONSTRUCTION.md` before changing anything under
`configs/benchmark.yaml`. It is the pre-registration for how the 75 records were
selected.

## The one rule that matters most

`MWLRDataCHR...csv` contains a coordinate for **every** description in the
corpus, including every benchmark and dev record. It is evaluation-only.

- Only `src/georef/data/benchmark_builder.py` may read it.
- The ChromaDB corpus must come from the LINZ gazetteer, ingested separately.
  Embedding this CSV would put the answer key inside the retrieval index.
- `tests/isolation/` enforces this and fails the build if a prediction-side
  module references it.

## Layout

```
configs/     pre-registered policy and paths
data/        source (git-ignored), benchmark, ground_truth, dev
docs/        BENCHMARK_CONSTRUCTION.md
scripts/     build_benchmark.py
src/georef/  normalise, config, provenance, data
tests/       unit, data, isolation, reproducibility
```

## Test categories

Tests are labelled by the evidence they provide, and the labels are not
interchangeable:

| Marker in docstring | Meaning |
|---|---|
| `UNIT TEST` | logic only, no data files |
| `REAL OFFLINE INTEGRATION TEST` | runs against the actual supplier data |
| `ISOLATION TEST` | proves a boundary cannot be crossed |
| `REPRODUCIBILITY TEST` | rebuild produces identical hashes |

Run the full suite with `python -m pytest -q`; skip the slow rebuild with
`python -m pytest -q -m "not slow"`.

## Next

Step 2 is the LINZ ingestion pipeline (API key required), then ChromaDB.
