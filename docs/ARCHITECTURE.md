# Codebase and Architecture

Every module, what it is for, and the public functions in it. Generated
alongside the code and kept current: a description of an architecture that has
drifted from the code is worse than none.

**Last updated:** after the held-out evaluation. Regenerated from the source
tree, so it describes the code as frozen at `0f97cf091a529cf6`.

---

## 1. The shape of the thing

This is a research experiment supported by software, not an application. The
structure follows one rule: **the model interprets, and Python decides.**

```
                       DESCRIPTION
                            |
                  +---------+---------+
                  |                   |
           CONDITION A          CONDITION B
                  |                   |
                LLM                  LLM  (interpretation only)
                  |                   |
            coordinates      SpatialInterpretation
                  |                   |
                  |          anchor variants
                  |                   |
                  |          retrieval: exact | folded | BM25 | dense
                  |                   |
                  |          RRF fusion
                  |                   |
                  |          deterministic disambiguation  -> may ABSTAIN
                  |                   |
                  |          LINZ anchor coordinate
                  |                   |
                  |          WGS84 geodesy (GeographicLib)
                  |                   |
                  |          coordinate sanity  -> may ABSTAIN
                  |                   |
                  +---------+---------+
                            |
                     CONDITION C  (deterministic combination)
                            |
                     ---- FREEZE ----
                            |
                    EVALUATION ONLY
                            |
                   held-out ground truth
                            |
                  WGS84 geodesic error, paired tests
```

The same model and the same decoding parameters run both conditions. A
difference there would measure model capability rather than the intervention.

## 2. Directory structure

```
dissertation-georef/
├── configs/                    pre-registered policy; committed and hashed
│   ├── benchmark.yaml          eligibility, quality gate, independence, sizes, seeds
│   ├── linz.yaml               layer, endpoint, field mapping, bounding boxes, retries
│   ├── models.yaml             LLM profiles, decoding parameters, embedder, reranker
│   ├── apis.yaml               base URLs, timeouts, workers, retry and jitter
│   ├── retrieval.yaml          channels, fusion, rerank, document template, hints
│   ├── disambiguation.yaml     scoring weights, abstention score and margin
│   ├── pipeline.yaml           multi-anchor policy, coordinate sanity boxes
│   ├── condition_c.yaml        combination rule: agreement @ 25 km (selected)
│   └── paths.yaml              source filenames and output directories
│
├── data/
│   ├── source/                 supplier files (git-ignored)
│   ├── benchmark/              75 records + manifest + hash  [SELECTION]
│   ├── ground_truth/           benchmark answers + audit report
│   ├── dev/                    100 records, descriptions and answers  [SELECTION]
│   ├── validation/             81 records  [HELD-OUT TEST -- read once]
│   ├── linz/                   raw, normalised snapshot, manifests
│   └── chroma/                 vector index + index manifest
│
├── docs/
│   ├── PLAN.md                 live status of the whole project
│   ├── RESULTS.md              what was measured, what it tested, what it concluded
│   ├── BENCHMARK_CONSTRUCTION.md   pre-registration of the data sets
│   ├── CONDITION_C_PREREGISTRATION.md  pre-registration of the third condition
│   ├── DATA_PROVENANCE.md      where each dataset came from, and what may read it
│   └── ARCHITECTURE.md         this file
│
├── experiments/                (generated; freeze_manifest.json is machine-local
│   │                            and git-ignored, since it hashes your snapshot
│   │                            and index)
│   ├── freeze_manifest.json    hashes of everything a prediction depends on
│   ├── baseline/<run_id>/      Condition A predictions + manifest
│   ├── hybrid/<run_id>/        Condition B predictions + manifest
│   ├── dev_*/ validation_*/    the same, per dataset
│   ├── combined/<id>/          Condition C outputs
│   └── evaluation/<id>/        metrics, statistics, paired case-level table
│
├── scripts/                    one entry point per stage; see section 4
├── src/georef/                 the library; see section 5
└── tests/                      unit, data, isolation, reproducibility
```

## 3. Data flow, and the boundary that matters

```
supplier files ──> benchmark_builder ──> benchmark / dev / validation
                                              │
LINZ WFS ──> snapshot (hashed) ──> ChromaDB index
                                              │
                                    ┌─────────┴─────────┐
                                    │ PREDICTION        │  never sees answers
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │ EVALUATION        │  loads ground truth
                                    └───────────────────┘
```

The corpus that supplied the descriptions also contains a coordinate for every
one of them. It is evaluation-only. `tests/isolation/` walks every module under
`src/` and every prediction-side script, and fails the build if any of them
references ground truth, the coordinate CSV, or its column names.

## 4. Scripts

| Script | Stage | Reads answers? |
|---|---|---|
| `build_benchmark.py` | Construct and hash the three data sets | Yes — it is the only builder |
| `fetch_linz.py` | Fetch and freeze the gazetteer snapshot | No |
| `verify_linz.py` | Report snapshot contents and anchor coverage | No |
| `build_index.py` | Embed the snapshot into ChromaDB | No |
| `check_llm.py` | Live smoke test of an endpoint | No |
| `extract_dev.py` | Interpretation over the dev descriptions | No |
| `ground_dev.py` | Offline grounding + scoring on dev | Yes — it is an evaluator |
| `freeze.py` | Hash every input; `--verify` detects drift | No |
| `run_benchmark.py` | Run a condition over a dataset | **No** — asserted by test |
| `evaluate.py` | Join predictions to ground truth, paired tests | Yes — the only place |
| `combine_conditions.py` | Derive Condition C from two finished runs | Yes — evaluator |

## 5. Modules

### `src/georef/config/loader.py`

Central configuration loading (master spec section 32).

- `repo_root()` — Repository root, resolved from this file's location.
- `load_yaml()` — Load a YAML file into a dict, failing loudly if it is absent or empty.
- `resolve_llm_profile()` — Return ``models_config`` with ``llm`` set to the chosen profile.
- `load_benchmark_config()` — —
- `load_paths_config()` — —

### `src/georef/data/benchmark_builder.py`

Construct and freeze the benchmark, ground truth and dev pool.

- **`BuildResult`** — Paths and counts produced by a build, for reporting and testing.  
  methods: —
- `compile_pattern()` — Compile the eligibility pattern, stripping the readability whitespace.
- `audit_description()` — Resolve one description against the coordinate corpus.
- `resolve_source_file()` — Find the configured file, tolerating separator and case differences.
- `build_benchmark()` — Run the full construction and write every artefact under ``output_root``.
- `main()` — —

### `src/georef/disambiguation/anchors.py`

Anchor strings, and the forms worth looking up.

- `anchor_variants()` — Query forms for an anchor, most literal first, without duplicates.
- `feature_terms()` — Feature types implied by generic words inside the anchor.
- `is_generic_only()` — True when the anchor names no specific place.
- `distinctive_tokens()` — Tokens that identify WHICH place, with generic terms removed.
- `name_similarity()` — Overlap of distinctive tokens, in [0, 1].
- `named_phrases()` — Capitalised multi-word phrases in a description, longest first.
- `resolve_back_reference()` — Resolve a generic anchor to the feature the description names.

### `src/georef/disambiguation/resolver.py`

Deterministic candidate selection and abstention (master spec sections 20–24).

- **`SelectionStatus`** — —  
  methods: —
- **`CandidateScore`** — —  
  methods: `to_provenance()`
- **`Selection`** — —  
  methods: `is_selected()`, `to_provenance()`
- **`Disambiguator`** — —  
  methods: `score_candidate()`, `select()`

### `src/georef/evaluation/metrics.py`

Positional error metrics (master spec sections 37 and 38).

- `geodesic_error_km()` — WGS84 geodesic distance between prediction and ground truth, in km.
- **`ErrorSummary`** — —  
  methods: `coverage()`, `to_dict()`
- `summarise()` — Summarise per-record results.

### `src/georef/evaluation/statistics.py`

Paired statistical comparison of the two conditions (master spec section 40).

- **`PairedSet`** — Records where both conditions produced a coordinate.  
  methods: `n_pairs()`, `denominators()`
- `pair_conditions()` — Split records by which conditions produced a usable coordinate.
- **`WilcoxonResult`** — —  
  methods: `to_dict()`
- `wilcoxon_signed_rank()` — Two-sided Wilcoxon signed-rank on paired errors (A minus B).
- **`McNemarResult`** — —  
  methods: `to_dict()`
- `mcnemar()` — Exact two-sided McNemar on paired binary outcomes.

### `src/georef/linz/client.py`

LINZ Data Service WFS client.

- **`LinzApiError`** — A LINZ request failed after the configured retries.  
  methods: —
- **`LinzAuthError`** — The key was rejected. Not retried: this is a configuration fault.  
  methods: —
- `redact()` — Replace any ``key=...`` value with ``key=REDACTED``.
- **`Page`** — One WFS page.  
  methods: —
- **`LinzClient`** — Paged reader for a single LDS WFS layer.  
  methods: `build_url()`, `fetch_page()`, `describe()`, `iter_features()`

### `src/georef/linz/ingest.py`

LINZ ingestion: raw fetch to frozen, hashed snapshot.

- **`IngestResult`** — —  
  methods: —
- `normalise_feature()` — Map one WFS feature onto the internal record schema.
- `ingest()` — Fetch the layer, validate and normalise it, and write a hashed snapshot.
- `main()` — —

### `src/georef/linz/verify.py`

Verification of a frozen LINZ snapshot (acceptance gate 2 and gate 5 evidence).

- `load_snapshot()` — —
- `summarise_snapshot()` — —
- `extract_anchor()` — Crude anchor extraction for coverage diagnostics only.
- `anchor_coverage()` — How many extracted anchors match a gazetteer name exactly after normalisation.
- `main()` — —

### `src/georef/llm/baseline.py`

Condition A: the standalone LLM baseline (master spec sections 4 and 64).

- **`BaselineStatus`** — —  
  methods: —
- **`BaselineCoordinates`** — —  
  methods: —
- `tool_parameters_schema()` — —
- **`BaselineResult`** — —  
  methods: `to_record()`
- **`BaselineGeoreferencer`** — —  
  methods: `predict()`

### `src/georef/llm/client.py`

LLM access through the official OpenAI SDK, pointed at an OpenAI-compatible

- **`LlmError`** — A completion failed after the configured retries.  
  methods: —
- **`LlmAuthError`** — Credentials were rejected or absent. Not retried.  
  methods: —
- **`LlmResponse`** — —  
  methods: `model_switched()`, `to_provenance()`
- **`LlmClient`** — Thin, frozen-parameter wrapper around the OpenAI chat completions API.  
  methods: `request_parameters()`, `complete_structured()`, `complete()`

### `src/georef/llm/extraction.py`

Semantic interpretation stage.

- **`ExtractionResult`** — An interpretation plus the LLM provenance that produced it.  
  methods: `to_record()`
- **`SpatialExtractor`** — —  
  methods: `extract()`

### `src/georef/models/spatial.py`

Typed spatial interpretation (master spec section 13).

- **`ExtractionStatus`** — Outcomes of the interpretation stage (master spec section 28).  
  methods: —
- **`SpatialClause`** — One distance-and-direction relation, exactly as written.  
  methods: —
- **`DerivedClause`** — A clause plus the deterministically computed bearing and distance.  
  methods: `is_usable()`
- **`ModelInterpretation`** — Exactly what the model reported. No derived values.  
  methods: —
- **`SpatialInterpretation`** — Model report plus deterministically derived quantities.  
  methods: `usable_clauses()`, `is_usable()`, `bearing_disagreement()`, `to_provenance()`
- `derive()` — Convert every clause, and decide whether anything usable came out.
- `tool_parameters_schema()` — JSON schema for the single forced tool call.

### `src/georef/normalise.py`

Deterministic text normalisation.

- `normalise()` — Return the canonical form of ``text`` used as a join and lookup key.
- `normalisation_version()` — Version identifier for the normalisation behaviour, for provenance.
- `fold_diacritics()` — Strip combining marks: 'Ōpōtiki' -> 'Opotiki', 'Whangamatā' -> 'Whangamata'.
- `match_key()` — Lookup key for matching a name against the gazetteer.
- `match_key_version()` — —

### `src/georef/pipeline/condition_b.py`

Condition B: the full hybrid pipeline (master spec section 65).

- **`ConditionBResult`** — —  
  methods: `to_record()`
- **`ConditionBPipeline`** — —  
  methods: `predict()`

### `src/georef/pipeline/condition_c.py`

Condition C: combining the baseline and the hybrid.

- **`CombinationResult`** — —  
  methods: `to_record()`
- `combine_record()` — Apply one combination rule to a single paired record.
- `combine()` — Apply a rule across every record, keyed by benchmark id.

### `src/georef/pipeline/grounding.py`

Grounding: turn an interpretation into a predicted coordinate.

- **`PredictionStatus`** — —  
  methods: —
- **`Prediction`** — —  
  methods: `is_success()`, `to_record()`
- **`Grounder`** — —  
  methods: `predict()`

### `src/georef/pipeline/runner.py`

Resumable, parallel execution over a set of records.

- **`JsonlCheckpoint`** — Append-only JSONL checkpoint keyed by a record ID field.  
  methods: `completed()`, `append()`, `finalise()`
- `run_parallel()` — Run ``work`` over ``items`` with bounded concurrency, checkpointing each result.

### `src/georef/provenance/freeze.py`

Experiment freeze and manifest (master spec sections 35 and 56).

- `build_freeze_manifest()` — Collect everything that must not change between freeze and benchmark.
- `freeze_hash()` — One digest over everything except the timestamp and the digest itself.
- `verify_against_freeze()` — Differences between the current state and a freeze manifest.

### `src/georef/provenance/hashing.py`

Hashing and environment capture for reproducibility (master spec section 34).

- `sha256_file()` — SHA-256 of a file's raw bytes.
- `sha256_text()` — SHA-256 of a UTF-8 encoded string.
- `environment_fingerprint()` — Software versions relevant to the reproducibility of data artefacts.

### `src/georef/retrieval/candidates.py`

Retrieval candidates and rank fusion.

- **`Candidate`** — One retrieved gazetteer record plus its retrieval provenance.  
  methods: `name()`, `feature_type()`, `land_district()`, `latitude()`, `longitude()`, `to_provenance()`
- `reciprocal_rank_fusion()` — Fuse ranked identifier lists into one ordering.

### `src/georef/retrieval/dense.py`

Dense retrieval channel: embedding the frozen LINZ snapshot into ChromaDB.

- **`IndexBuildResult`** — —  
  methods: —
- **`SentenceTransformerEmbedder`** — Wraps sentence-transformers. Constructed only by the build script.  
  methods: `dimension()`, `encode()`
- **`ChromaVectorStore`** — Persistent ChromaDB collection keyed by LINZ identifier.  
  methods: `reset()`, `add()`, `count()`, `query()`
- `build_index()` — Embed every snapshot record and load it into the vector store.
- `load_snapshot()` — —

### `src/georef/retrieval/hybrid.py`

Hybrid retrieval: four channels, rank fusion, optional cross-encoder rerank.

- **`Embedder`** — —  
  methods: `encode()`
- **`VectorStore`** — —  
  methods: `query()`
- **`Reranker`** — —  
  methods: `score()`
- **`HybridRetriever`** — —  
  methods: `retrieve()`

### `src/georef/retrieval/lexical.py`

Lexical retrieval channels.

- `tokenise()` — Lowercase, diacritic-folded alphanumeric tokens.
- **`LexicalIndex`** — In-memory exact, folded and BM25 indexes over the gazetteer snapshot.  
  methods: `document_text()`, `exact_name()`, `folded_name()`, `bm25()`

### `src/georef/retrieval/name_spotter.py`

Spotting gazetteer names inside a description, before interpretation.

- **`NameSpotter`** — Finds gazetteer names occurring in a description.  
  methods: `spot()`, `spot_records()`

### `src/georef/retrieval/rerank.py`

Cross-encoder reranking.

- **`CrossEncoderReranker`** — Scores (query, document) pairs with a sentence-transformers CrossEncoder.  
  methods: `score()`

### `src/georef/spatial/bounds.py`

Bounding-box containment, shared by the benchmark builder and LINZ ingest.

- `containing_box()` — Name of the first box containing the point, or ``None``.
- `is_inside_any()` — —

### `src/georef/spatial/geodesic.py`

Deterministic WGS84 geodesic computation (master spec section 26).

- **`GeodesicResult`** — —  
  methods: `as_tuple()`
- `destination()` — Solve the direct geodesic problem on the WGS84 ellipsoid.
- `distance_between()` — Solve the inverse problem: geodesic distance in metres.
- `bearing_between()` — Initial bearing from point 1 to point 2, in degrees clockwise from north.
- `interpolate()` — Point at ``fraction`` of the way along the geodesic from 1 to 2.

### `src/georef/spatial/resolution.py`

Turning one or more grounded clauses into a single predicted coordinate.

- **`ResolutionStatus`** — —  
  methods: —
- **`GroundedClause`** — A clause whose anchor has been resolved to authoritative coordinates.  
  methods: `destination()`
- **`Resolution`** — —  
  methods: `is_success()`, `to_provenance()`
- `resolve_clauses()` — Produce one predicted coordinate from the grounded clauses.

### `src/georef/spatial/units.py`

Deterministic conversion of direction text to bearings, and distances to metres.

- **`UnknownDirection`** — The direction text has no defensible bearing.  
  methods: —
- **`UnknownUnit`** — The distance unit is not one we convert.  
  methods: —
- `bearing_for()` — Bearing in degrees clockwise from true north.
- `distance_in_metres()` — Convert a distance to metres. Rejects missing, negative and unknown units.

## 6. Design decisions embedded in the structure

**The model never computes.** Bearings come from a lookup table and distances
from a conversion table, both in `spatial/units.py`. The prompt tells the model
to report `"southwest"` and `3.5 miles`, never `225` and never kilometres. Any
bearing the model volunteers is recorded and ignored, with the disagreement
reported.

**Retrieval proposes, disambiguation disposes.** With 41,402 distinct names
across 49,771 records, one name in six is shared, so a retrieval channel that
returned a single hit would be making the decision. Every channel returns a
list.

**Abstention is a first-class outcome.** Nine named statuses, never collapsed
into a generic error. A prediction outside New Zealand is rejected rather than
plotted.

**Nothing is tuned on held-out data.** Selection happens on the selection set;
the held-out set is read once. `freeze.py --verify` makes drift detectable and
`run_benchmark.py` refuses to run against a drifted configuration.

**Provenance travels with every record.** Requested and served model, provider,
profile, attempt count, raw response, structured response, every candidate
considered with its individual feature scores, the score margin, the context
names used, and the geodesic method.

## 7. Test categories

| Label | Meaning |
|---|---|
| `UNIT TEST` | Logic only. Fakes for network, models, databases |
| `REAL OFFLINE INTEGRATION TEST` | Runs against the actual supplier data |
| `LIVE API SMOKE TEST` | Hits a real external service |
| `ISOLATION TEST` | Proves a boundary cannot be crossed |
| `REPRODUCIBILITY TEST` | A rebuild produces identical hashes |

A mocked test never licenses a claim that the real thing works. The labels are
not interchangeable and the distinction is enforced in how results are reported.
