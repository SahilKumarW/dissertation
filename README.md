# dissertation-georef

MSc Artificial Intelligence dissertation: natural-language spatial
georeferencing. A controlled A/B comparison between a standalone LLM baseline
(**Condition A**) and a retrieval-grounded, LINZ-grounded, deterministic-GIS
hybrid pipeline (**Condition B**), evaluated on paired WGS84 geodesic positional
error over 75 New Zealand benchmark descriptions.

| Document | What it is |
|---|---|
| This file | How to run it |
| `docs/PLAN.md` | Live status of the whole project |
| `docs/RESULTS.md` | What has been measured, what each measurement tested, and what was concluded |
| `docs/ARCHITECTURE.md` | Every module, what it does, and the public functions in it |
| `docs/BENCHMARK_CONSTRUCTION.md` | The pre-registration an examiner would read first |
| `docs/CONDITION_C_PREREGISTRATION.md` | Condition C, its hypotheses, and the protocol revision |
| `docs/DATA_PROVENANCE.md` | Where every dataset came from and what may read it |

**Status: complete.** The experiment has been frozen
(`0f97cf091a529cf6`, v2), run, and evaluated on 81 held-out records. The result
is in `docs/RESULTS.md` section 10: the hybrid does not improve average
positional accuracy over the standalone baseline, a null that replicates on two
independent sets.

Everything below reproduces that from scratch.

---

## Install

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install sentence-transformers chromadb bm25s
```

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install sentence-transformers chromadb bm25s
```

PowerShell uses `.venv\Scripts\Activate.ps1`; if script execution is blocked,
run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

`sentence-transformers` pulls PyTorch, which is a large download.

## Credentials

Copy `.env.example` to `.env` and fill in. **Secrets only** — model identifiers,
base URLs and decoding parameters live in `configs/`, committed and hashed, so
they appear in the provenance record.

```
NVIDIA_API_KEY=nvapi-...
LINZ_API_KEY=...
GOOGLE_MAPS_API_KEY=
```

- **NVIDIA** — https://build.nvidia.com, serves `openai/gpt-oss-20b`
- **LINZ** — https://data.linz.govt.nz/my/api/, needed once for the snapshot
- **Google** — not used until step 6; leave blank

`.env` is git-ignored and must stay that way.

## The one rule that matters most

`MWLR-Data-CHR-...csv` holds a coordinate for **every** description in the
corpus, benchmark and dev records included. It is evaluation-only.

- Only `georef.data.benchmark_builder` may read it
- The ChromaDB corpus comes from the LINZ gazetteer, ingested separately;
  embedding the MWLR file would put the answer key inside the retrieval index
- `tests/isolation/` scans every module under `src/` and fails the build if a
  prediction-side file references it

---

## Running it, step by step

Each step is independent and re-runnable. Later steps read the frozen artefacts
of earlier ones, never the live sources.

### Step 1 — Build the benchmark

```bash
python scripts/build_benchmark.py
```

Reads the two supplier files from `data/source/` and writes:

| Output | Contents |
|---|---|
| `data/benchmark/benchmark.csv` | 75 records: `benchmark_id`, `description`. **No coordinates** |
| `data/ground_truth/ground_truth.csv` | Coordinates, audit category, source event IDs |
| `data/dev/dev_ground_truth.csv` | 100 records for configuration selection |
| `data/dev/dev_pool.csv` | 3,945 descriptions, no coordinates |
| `data/benchmark/benchmark_manifest.json` | Policy, counts, artefact/source/code hashes |

Filenames need not match `configs/paths.yaml` exactly — copies differing only in
hyphens, spaces, underscores or case resolve, and the manifest records the name
actually read.

**Expected output.** 308 candidates; audit 179 / 108 / 0 / 21 / 0 / 0; 256
independent eligible; 75 benchmark; 100 dev.

**Expected hashes (`nz-dd-v2`).**

```
benchmark.csv         50d2759cc374350ed278b53b24b5e91e1bf498f56cfc57e01ad47ea2777cb95a
ground_truth.csv      4e42f42c718ba41138eeeeb89286dbe8af7b1364a81130841d2ca09927f89a4d
dev_ground_truth.csv  3bb06a95f516cc1457f46d9090acdda03809c906c89586246abb5789611241a3
```

Verified identical on Windows / Python 3.13 / pandas 3.0.5 and Linux /
Python 3.12 / pandas 3.0.2.

### Step 2 — Freeze the LINZ snapshot

```bash
python scripts/fetch_linz.py --describe   # verify key, layer and field mapping
python scripts/fetch_linz.py              # full ingest
python scripts/verify_linz.py             # what you actually got
```

`--describe` costs one small request and prints the field names the service
returns. Check them against `fields:` in `configs/linz.yaml` before the full
fetch, so a renamed column fails loudly rather than emptying the snapshot.

**LINZ is called once.** Every later stage reads the hashed snapshot, so a LINZ
update mid-project cannot change your results.

**Expected output.** 49,771 records kept, 4,957 rejected as
`OUTSIDE_BOUNDING_BOXES` (the layer covers the Ross Dependency and undersea
features), zero rejected for missing fields, bad coordinates or duplicate IDs.

`verify_linz.py` reports feature types, administrative-field coverage and
geometry types, then estimates how many description anchors exist in the
gazetteer. That anchor check is a crude regex diagnostic, not the pipeline, and
runs against the **dev pool only** — measuring it on the benchmark would mean
inspecting the evaluation set before the experiment.

### Step 3 — Build the index

```bash
python scripts/build_index.py
```

Embeds every snapshot record with `all-MiniLM-L6-v2` and loads ChromaDB.

The collection is dropped and recreated each time: an incremental add would
leave records from a superseded snapshot behind, and a candidate traced to the
wrong snapshot is a provenance failure nothing downstream could detect. If the
loaded count does not equal the snapshot count, the build raises.

**Expected output.** 49,771 records, 384 dimensions, collection
`nz_geographic_entities`, plus `data/chroma/index_manifest.json` tying the index
to the snapshot hash.

### Step 4 — Interpret the dev descriptions

```bash
python scripts/extract_dev.py --limit 10          # pilot
python scripts/extract_dev.py                     # full 100, resumes
python scripts/extract_dev.py --retry-failures    # re-request only failures
```

**Switching endpoints.** `--profile` selects an entry from `llm_profiles` in
`configs/models.yaml`, and a provider name works as shorthand:

```bash
python scripts/check_llm.py --profile gemini      # is it up?
python scripts/extract_dev.py --profile gemini    # run on Gemini instead
python scripts/extract_dev.py --profile gemini_flash_lite
```

| Profile | Model | Notes |
|---|---|---|
| `nvidia_gpt_oss` | `openai/gpt-oss-20b` | Free tier, throttles under load |
| `gemini_flash` | `gemini-3.8-flash` | GA since 2 Sep 2026, 1M context |
| `gemini_flash_lite` | `gemini-3.5-flash-lite` | Cheaper and faster fallback |

Both providers go through the same OpenAI SDK; Google exposes an
OpenAI-compatible surface, so only the base URL, key and model id differ. Each
profile carries its own parameters: the Gemini profiles omit `seed` and
`reasoning_effort` because that endpoint rejects parameters it does not
implement, and a 400 is deliberately not retried.

Resume is by record id, so you can start on one provider, switch, and finish on
the other without duplicating work.

**Two-pass retrieval (ablation arm).** `--hints` queries the gazetteer with the
raw description first and tells the model which names in it are real New Zealand
places. **Names only, never coordinates**, so the model still does not know
where anything is. Off by default: it is a variant to be measured, not part of
the pipeline.

```bash
python scripts/extract_dev.py --profile gemini --hints \
    && move data\dev\interpretations.jsonl data\dev\interpretations_hints.jsonl
python scripts/ground_dev.py --interpretations data/dev/interpretations_hints.jsonl \
    --output data/dev/dev_predictions_hints.jsonl
```

**A caveat that matters for the experiment.** Profiles are a development
convenience. The frozen benchmark must run BOTH conditions on ONE profile — a
different model in each arm would measure model capability rather than the
effect of retrieval and deterministic GIS. Every record carries its provider,
profile and served model, and `extract_dev.py` prints a warning if one output
file mixes them.

Sends each description to `gpt-oss-20b` with one forced, schema-shaped tool
call. Bearings and unit conversions are done in Python from lookup tables, never
by the model.

Resume is by record ID: a completed record is never re-requested and never
overwritten, so an interrupted run costs nothing. The output is rewritten sorted
by ID when a run finishes, so the artefact does not depend on thread scheduling.
Add `--retry-failures` to re-request only records whose previous attempt failed.

**Expected output.** `data/dev/interpretations.jsonl`, one object per record,
carrying the raw response, the structured response, the served model, attempt
count and per-record latency.

**Observed.** Four workers: no retries, about 22 s per record, roughly nine
minutes for 100. Eight workers is worse, not better — 45 of 83 successes needed
a retry, 17 records were lost to dropped connections, and the run took 27
minutes. The default of 4 in `configs/apis.yaml` is measured, not cautious.

Changing the worker count changes wall-clock time and nothing else about the
result: records are independent. The run summary reports how many records needed
a retry, which is the signal to lower concurrency.

### Step 5 — Geodesy (no script; used by the pipeline)

`georef.spatial.geodesic` solves the WGS84 direct and inverse problems through
GeographicLib. `georef.spatial.resolution` turns one or more grounded clauses
into a single predicted coordinate under one of three policies, configured in
`configs/pipeline.yaml`.

Nothing to run on its own — exercised by `python -m pytest tests/unit/test_geodesic.py
tests/unit/test_resolution.py -q`, which checks it against published WGS84
constants, round-trips the direct and inverse problems, and covers the
antimeridian and both hemispheres.

### Step 6 — Disambiguation (no script; used by the pipeline)

`georef.disambiguation.anchors` produces the query forms for an anchor;
`georef.disambiguation.resolver` scores retrieved candidates and either selects
one or abstains, under the policy in `configs/disambiguation.yaml`.

Exercised by `python -m pytest tests/unit/test_anchors.py
tests/unit/test_disambiguation.py -q`. Every anchor string in those tests is a
real one from the 100-record dev run.

### Step 7 — Ground the interpretations you already have

```bash
python scripts/ground_dev.py                   # about 1 minute
python scripts/ground_dev.py --policy shortest # compare multi-anchor policies
python scripts/ground_dev.py --no-dense        # lexical channels only
python scripts/ground_dev.py --rerank          # add the cross-encoder, much slower
```

**Runtime.** About a minute for 100 records: roughly 20 s to load the snapshot
and build the BM25 index, a few seconds for the embedder and Chroma, then well
under a second per record. Reranking is opt-in because the model is a ~2.3 GB
download and adds a second or more per query on CPU, taking the run to several
minutes. Whether it earns that is one of the questions the dev set answers.

Runs retrieval, disambiguation, deterministic geodesy and coordinate sanity over
`data/dev/interpretations.jsonl`, then joins to the dev ground truth and reports
geodesic error. **No network and no model** — only the frozen snapshot and the
index — so it works on a partial set of interpretations and while the endpoint
is unavailable.

Writes `data/dev/dev_predictions.jsonl` with the full per-record provenance:
which candidates were considered, the feature scores behind the selection, the
score margin, the clause estimates and their disagreement.

This is the **dev** set, whose declared purpose is choosing the configuration
once before the freeze. No headline result comes from it.

### Step 8 — Freeze

```bash
python scripts/freeze.py            # write the freeze manifest
python scripts/freeze.py --verify   # confirm nothing has changed since
```

Hashes every input to a prediction: benchmark, ground truth, LINZ snapshot,
every config file, every source module, both prompts, and the index manifest.
`--verify` reports exactly which of them changed, so configuration drift is
caught before it produces results rather than after.

After freezing, no benchmark-driven tuning. Any change needs a new experiment
version.

**The freeze manifest is machine-specific** and is not distributed: it hashes
your LINZ snapshot and your ChromaDB index, which are built locally. Generate it
on the machine that will run the experiment, and verify there too.

### Step 9 — Run the benchmark

```bash
python scripts/run_benchmark.py --condition A
python scripts/run_benchmark.py --condition B
python scripts/run_benchmark.py --condition both --profile gemini
```

Both conditions see the same 75 descriptions and the same model. **This script
cannot read ground truth**, and a test asserts that.

It refuses to run against a changed configuration unless `--allow-unfrozen` is
given, in which case the manifest records that the run was not the frozen
experiment. Resumable by record id; `--limit` marks the run as a pilot in its
manifest.

Output goes to `experiments/baseline/<run_id>/` and
`experiments/hybrid/<run_id>/`, each with `predictions.jsonl` and a
`manifest.json` recording the freeze hash, the served models and the
configuration.

### Step 10 — Evaluate

```bash
python scripts/evaluate.py --baseline <run_id> --hybrid <run_id>
```

**Ground truth is loaded here and nowhere else.** This script reads finished
prediction files and never invokes a model, so it cannot influence what it
measures.

Writes to `experiments/evaluation/<id>/`:

| File | Contents |
|---|---|
| `metrics.json` | Per-condition summary with every denominator stated |
| `statistics.json` | Wilcoxon signed-rank and McNemar, paired |
| `paired_results.csv` | Case-level table for the qualitative error analysis |

Paired tests run over records where **both** conditions produced a coordinate.
That denominator is smaller than 75 and is reported alongside the result.

### Condition C — combination rules

```bash
# 1. Select the rule on the pooled selection set (dev + benchmark, 175 records)
python scripts/combine_conditions.py --sweep \
    --pair dev:<devA>:<devB> --pair benchmark:<benchA>:<benchB>

# 2. Write the winner into configs/condition_c.yaml, then freeze
python scripts/freeze.py --experiment-version v2

# 3. Run both conditions on the held-out set
python scripts/run_benchmark.py --condition both --dataset validation --profile gemini

# 4. Derive C and evaluate. ONE PASS.
python scripts/combine_conditions.py --dataset validation \
    --baseline <valA> --hybrid <valB>
python scripts/evaluate.py --baseline <valA> --hybrid <valB>
```

`--sweep` searches only the grids fixed in `configs/condition_c.yaml`, so the
search space cannot grow once numbers are in view. Step 4 refuses to run until
a strategy is set.

**Data set roles.** `dev` and `benchmark` are selection data; `validation` is
the held-out test and is read once. This is a declared protocol revision —
see `docs/CONDITION_C_PREREGISTRATION.md` section 2a.

---

## Tests

```bash
python -m pytest -q                # everything
python -m pytest -q -m "not slow"  # skip the full rebuild
```

Tests are labelled by the evidence they provide, and the labels are not
interchangeable:

| Label | Meaning |
|---|---|
| `UNIT TEST` | Logic only. Fakes for network, models and databases |
| `REAL OFFLINE INTEGRATION TEST` | Runs against the actual supplier data |
| `LIVE API SMOKE TEST` | Hits a real external service |
| `ISOLATION TEST` | Proves a boundary cannot be crossed |
| `REPRODUCIBILITY TEST` | A rebuild produces identical hashes |

A mocked test never licenses a claim that the real thing works.
`scripts/check_llm.py` and `scripts/fetch_linz.py --describe` are the live smoke
tests.

## Layout

```
configs/     pre-registered policy: benchmark, linz, models, apis, retrieval, paths
data/        source (git-ignored), benchmark, ground_truth, dev, linz, chroma
docs/        PLAN, BENCHMARK_CONSTRUCTION, DATA_PROVENANCE
scripts/     build_benchmark, fetch_linz, verify_linz, build_index, extract_dev, check_llm
src/georef/  config, normalise, provenance, spatial, data, linz, llm, models, retrieval, pipeline
tests/       unit, data, isolation, reproducibility
```

## Configuration

Everything that defines the experiment is in `configs/`, committed and hashed.
Production code reads it; nothing is duplicated in source.

| File | Holds |
|---|---|
| `benchmark.yaml` | Eligibility pattern, quality gate, independence rule, sizes, seeds |
| `linz.yaml` | Layer, endpoint, field mapping, bounding boxes, retry policy |
| `models.yaml` | LLM, decoding parameters, embedding model, reranker |
| `apis.yaml` | Base URLs, timeouts, workers, retry and jitter policy |
| `retrieval.yaml` | Channels, fusion method, rerank, document template, two-pass hints |
| `pipeline.yaml` | Multi-anchor policy, disagreement threshold, coordinate sanity boxes |
| `disambiguation.yaml` | Candidate scoring weights, abstention score and margin |
| `paths.yaml` | Source filenames and output directories |

Changing any of these creates a new experiment version. After the freeze, no
change is permitted at all.
