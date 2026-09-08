# Benchmark Construction — Pre-Registration

**Benchmark version:** `nz-dd-v1`
**Status:** policy fixed; artefacts built and hashed
**Applies to:** `configs/benchmark.yaml`, `src/georef/data/benchmark_builder.py`

This document records how the benchmark, ground truth and dev pool were
constructed, and why each decision was made. It is written to be read by an
examiner asking whether the benchmark could have been shaped to favour a
result.

---

## 1. Why the benchmark had to be constructed

The master specification assumes a supervisor-provided Jones workbook holding
75 New Zealand records, together with a ground-truth audit reporting
`SINGLE_ROW_EXACT 38`, `DUPLICATE_ROWS_SAME_COORDINATE 19`,
`SINGLE_ROW_NORMALIZED 1`, `UNRESOLVED 12`, `NO_MATCH 5`, and a defensible
denominator of 58.

That workbook is not available. Only two supplier files exist:

| File | Contents |
|---|---|
| `SampleDataNoCoords.xlsx` | 5,103 natural-language descriptions, no coordinates |
| `MWLRDataCHR20230307_nz_only_with_coordinates_desc_60_or_more.csv` | 30,461 New Zealand records with `verbatimLocality` and decimal coordinates |

**Consequence for the dissertation.** The audit figures above cannot be
reproduced and must not be cited. They describe a different artefact. The
benchmark used here was constructed by the procedure below and has its own
audit, reported in `data/ground_truth/audit_report.json`. This is a declared
deviation from the specification, not a silent substitution.

## 2. The leakage hazard this creates

Every one of the 5,103 descriptions appears verbatim as a `verbatimLocality`
value in the coordinate CSV. Overlap is 100%. That CSV therefore contains the
answer to every benchmark record *and* every dev record.

Three consequences are binding on the whole project:

1. The coordinate CSV is **evaluation-only**. `src/georef/data/benchmark_builder.py`
   is the only module permitted to read it, and it runs before any prediction
   code exists.
2. The ChromaDB corpus must be built from the LINZ gazetteer, ingested
   independently. Embedding the MWLR CSV would place the answer key inside the
   retrieval index and invalidate the experiment.
3. `tests/isolation/test_no_ground_truth_leakage.py` scans every module under
   `src/` and fails on any new prediction-side file that references
   `ground_truth`, `MWLRDataCHR`, `decimalLatitude`, `decimalLongitude`,
   `verbatimLocality` or `Jones`.

## 3. Procedure

Executed by `scripts/build_benchmark.py`. Every parameter lives in
`configs/benchmark.yaml`; none is embedded in code.

### 3.1 Normalisation

`src/georef/normalise.py`, shared with the prediction pipeline so a description
is keyed identically wherever it is handled: NFKC, ASCII apostrophes, quotes and
dashes, whitespace collapse, lowercase, trailing-punctuation strip.

Expansion of geographic abbreviations (`Mt` → `Mount`, `Co.` → `County`) is
deliberately **not** applied here. Those transformations alter meaning-bearing
tokens and would change which source rows a description joins to. They belong to
the prediction-time normaliser, where they can be tested against interpretation
behaviour.

### 3.2 Deduplication of the description corpus

5,103 rows reduce to 4,020 unique normalised descriptions. The first occurrence
in file order is retained, so the original text kept for a given key is
deterministic.

### 3.3 Eligibility: DISTANCE_DIRECTION

A description is a candidate if the pattern in
`configs/benchmark.yaml:distance_direction_pattern` matches: an optional hedge
(`ca`, `approx`, `about`, `~`), a number, a distance unit, a compass direction,
and `of` or `from`.

The pattern covers eight- and sixteen-point directions in spelled and
abbreviated form, with or without periods (`SE`, `S.E.`, `south-east`,
`east-northeast`, `N.N.W.`), and the units `km`, `m`, `mile`, `mls`.

313 of 4,020 descriptions match.

**Known limitation.** This is a sampling frame, not a classifier. It excludes
relations that are not compass-bearing offsets, which is intended
(`44.3 km upstream of Pipiriki`, `along the 9.2 km section of road`). It will
also miss unusual phrasings. Recall of the frame does not affect the validity of
the paired comparison, because both conditions see the identical records.

### 3.4 Ground-truth audit

Each candidate is resolved against the coordinate corpus, exact match on the
untouched `verbatimLocality` first, normalised match as a labelled fallback:

| Category | Meaning | Eligible |
|---|---|---|
| `SINGLE_ROW_EXACT` | one source row, exact match | yes |
| `DUPLICATE_ROWS_SAME_COORDINATE` | several rows, one coordinate | yes |
| `SINGLE_ROW_NORMALIZED` | resolved only after normalisation | yes |
| `UNRESOLVED_MULTI_COORD` | several rows, conflicting coordinates | no |
| `NO_COORDINATE` | matched rows carry no coordinate | no |
| `NO_MATCH` | no matching row | no |

Coordinates are compared after rounding to 6 decimal places (~0.11 m), which
collapses float noise without merging distinct points.

Result: 184 / 108 / 0 / 21 / 0 / 0. `SINGLE_ROW_NORMALIZED` is zero and
`NO_MATCH` is zero because the descriptions were extracted verbatim from the
corpus; both categories are retained so that a future corpus change is visible.

### 3.5 Quality gate

A record is eligible only if its category is eligible, its coordinate is
present and numeric, and the point falls inside one of two bounding boxes.

The **Chatham Islands** need their own box: they sit east of the antimeridian
and carry negative longitudes, so a single naive box would silently discard
them. One Chatham record is present in the corpus and survives the gate.

**Fields deliberately not used as gates.** `coordinateUncertaintyInMeters` is
populated on 10 of 30,461 rows and `validationStatus` on 787; neither supports a
filter. `georeferenceIsDerived` is fully populated, but its meaning is not
documented in the supplied extract, so gating on it would be a guess presented
as a criterion. Both `georeferenceIsDerived` and `verbatimCoordinateSystem` are
carried into `ground_truth.csv` as descriptive metadata for the error analysis
and for any sensitivity check.

### 3.6 Item independence

Some descriptions are near-duplicates of the same site recorded twice
(`Tongariro National Park, 0.5km E of Erua township, western slope of
Hauhungatahi` and `Tongariro National Park, W slope of Mt Hauhungatahi, 0.5km E
of Erua township`). They resolve to the identical coordinate and are not
independent items: a paired test would count the same evidence twice.

At most one record is kept per distinct ground-truth coordinate, chosen as the
first by sorted normalised key. This removed 31 records, leaving 261 independent
eligible records.

### 3.7 Selection

75 records sampled with `random.Random(20260908).sample` from the eligible keys
in sorted order, then re-sorted and numbered `NZDD-001` … `NZDD-075`.

Selection depends only on this document's policy. It does not depend on any
model output, because no model existed when the benchmark was built.

`n = 75` matches the size in the specification. 261 eligible records remain
available, so a larger benchmark is possible if more statistical power is
wanted; that would be a new benchmark version, not an edit to this one.

## 4. Denominator

All 75 selected records have a single defensible coordinate, so the defensible
ground-truth denominator is **75**, and it is not the same quantity as the 58
quoted in the specification, which belonged to the Jones workbook.

Prediction coverage is a separate denominator: a condition may abstain, and the
metrics report abstentions explicitly rather than dropping them.

## 5. Ground-truth precision, an honest limit

Ground truth is the georeferenced coordinate the original collector assigned to
the whole locality string, not a coordinate derived from the distance-direction
clause alone. Spot checks against hand-entered anchor coordinates show the
answer key reflects the described offset rather than the anchor town:

| Record | Stated | Measured from anchor |
|---|---|---|
| `54 km ESE of New Plymouth` | 54 km, ESE (~112°) | 58.2 km at 102° |
| `8 km NE of Akaroa` | 8 km, NE (45°) | 7.8 km at 30° |
| `3 km S of Twizel` | 3 km, S (180°) | 2.3 km at 191° |
| `8 km north of Waiouru` | 8 km, N (0°) | 13.6 km at 26° |

Agreement is good but not exact, and the last row shows it can be several
kilometres out. Most source coordinates originate from NZMS 260 map-sheet grid
references rather than direct GPS, so a positional error floor of roughly one to
a few kilometres is inherent to the benchmark. **No system can be shown to beat
that floor**, and any claim of sub-kilometre accuracy would be an artefact of
the answer key rather than evidence about the method. The floor applies equally
to both conditions, so the paired comparison remains valid.

## 6. Artefacts

| Path | Contents |
|---|---|
| `data/benchmark/benchmark.csv` | `benchmark_id`, `description` — **no coordinates** |
| `data/benchmark/benchmark_manifest.json` | policy, counts, artefact/source/code hashes, environment |
| `data/benchmark/benchmark_hash.txt` | SHA-256 of `benchmark.csv` |
| `data/ground_truth/ground_truth.csv` | coordinates, audit category, source event IDs, metadata |
| `data/ground_truth/ground_truth_hash.txt` | SHA-256 of `ground_truth.csv` |
| `data/ground_truth/audit_report.json` | full audit counts |
| `data/dev/dev_pool.csv` | 3,945 non-benchmark descriptions, **no coordinates** |

The manifest records `generated_at_utc`, so it is not byte-reproducible.
Reproducibility is asserted over the three artefact hashes, which are stable for
a fixed policy, source pair and interpreter, and is tested.

## 7. Change control

Any change to `configs/benchmark.yaml` produces a different benchmark. Such a
change requires a new `benchmark_version`, a new build, and an entry in this
document stating what changed and why. After the experiment freeze, no change is
permitted at all.
