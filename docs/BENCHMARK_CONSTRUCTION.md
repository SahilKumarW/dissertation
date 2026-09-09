# Benchmark Construction — Pre-Registration

**Benchmark version:** `nz-dd-v2` — see the change log in section 7 for what
superseded `nz-dd-v1` and why
**Status:** frozen. Artefacts built, hashed and used; the experiment is complete
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

308 of 4,020 descriptions match.

The `\b` after the unit group is load-bearing; see the change log for what it
prevents.

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

Result: 179 / 108 / 0 / 21 / 0 / 0. `SINGLE_ROW_NORMALIZED` is zero and
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
first by sorted normalised key. This removed 31 records, leaving 256 independent
eligible records.

### 3.7 Selection

75 records selected by ordering the eligible keys on `sha256("20260908:<key>")`
and taking the first 75, then re-sorted and numbered `NZDD-001` … `NZDD-075`.

Selection depends only on this document's policy. It does not depend on any
model output, because no model existed when the benchmark was built.

`n = 75` matches the size in the specification. 256 eligible records remain
available, so a larger benchmark is possible if more statistical power is
wanted; that would be a new benchmark version, not an edit to this one.

### 3.8 Development ground truth

100 records selected by the same hash ordering under seed `20260909`, from the
eligible records the benchmark did not take, numbered `DEVGT-001` … `DEVGT-100`.

Its purpose is declared here in advance and is bounded: **selecting the
configuration of Condition B once, before the experiment freeze.** Concretely
that covers

* the extraction prompt version;
* which retrieval channels are enabled, and whether cross-encoder reranking is
  used;
* the multi-anchor resolution policy and its disagreement threshold;
* the candidate-disambiguation weights and the score-margin abstention
  threshold.

It exists so those choices are measured rather than asserted. It does not cover
anything measured against the benchmark, and it is used once: a configuration is
chosen, the experiment is frozen, and the dev set is not consulted again.

This is a widening of the purpose recorded when the set was first built, which
named retrieval configuration only. The widening is recorded here, with its
date, rather than applied silently — a pre-registration that quietly grows to
cover whatever was done to it is not a pre-registration. It was made before any
prediction was run against the benchmark and before the freeze.

It is not a second benchmark. No headline result is reported from it, and after
the freeze it is not consulted again. Tests assert that no dev description
appears in the benchmark and that no dev record shares a ground-truth
coordinate with a benchmark record, so the independence rule spans both sets.

Adding it does not disturb the benchmark: the benchmark is drawn first from the
sorted eligible keys, and the dev set is drawn from the remainder with its own
seed. Both published hashes are unchanged, and a test asserts it.

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

## 7. Change log

### nz-dd-v2 (supersedes nz-dd-v1, before any experiment was run)

Two corrections, both made before the freeze and before any prediction existed.

**Eligibility defect.** The v1 pattern matched `14 miles from Glade House` by
consuming the plural `s` of "miles" as a South bearing: unit `mile`, direction
`s`, then `from`. Five descriptions stating no direction at all were admitted
as DISTANCE_DIRECTION. It surfaced when the extraction pilot returned
`RELATIVE_LOCATION` for one of them and the pipeline correctly abstained.

The fix is a word boundary immediately after the unit group, so a unit's own
plural cannot be read as a direction. Candidates fall from 313 to 308,
independent eligible records from 261 to 256. Every true positive still
matches, including `1 mile Sth of Minaret Creek` and `55 mls N.N.W. of
Auckland`, and there are tests for both directions of this.

**Selection fragility.** Correcting five ineligible records changed 52 of the
75 selections, because `random.sample` draws as a function of the whole list.
Selection now orders eligible keys by `sha256("<seed>:<key>")` and takes the
first N. It is still deterministic and seed-controlled, but each record's
position depends only on itself, so a future construction fix disturbs only the
records it actually concerns. Verified: shrinking the benchmark from 75 to 70
yields a nested subset of exactly 70 of the original records.

SHA-256 ordering is also portable in a way the Mersenne Twister is not — the
draw is identical on any interpreter and platform.

**v2 hashes.**

    benchmark.csv         50d2759cc374350ed278b53b24b5e91e1bf498f56cfc57e01ad47ea2777cb95a
    ground_truth.csv      4e42f42c718ba41138eeeeb89286dbe8af7b1364a81130841d2ca09927f89a4d
    dev_ground_truth.csv  3bb06a95f516cc1457f46d9090acdda03809c906c89586246abb5789611241a3

The v1 hashes quoted earlier in this document and in `DATA_PROVENANCE.md` are
superseded and must not be cited.

## 8. Change control

Any change to `configs/benchmark.yaml` produces a different benchmark. Such a
change requires a new `benchmark_version`, a new build, and an entry in this
document stating what changed and why. After the experiment freeze, no change is
permitted at all.
