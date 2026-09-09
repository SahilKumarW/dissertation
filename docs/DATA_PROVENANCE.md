# Data Provenance

Every dataset the experiment depends on is frozen and hashed before use. This
document says where each came from, what may read it, and how to verify it.

## 1. The three data sets

Constructed from the two supplier files by one deterministic, hash-ordered
procedure. Full pre-registration in `BENCHMARK_CONSTRUCTION.md`; the roles
below reflect the protocol revision recorded in
`CONDITION_C_PREREGISTRATION.md` section 2a.

| Set | n | Role | Ground-truth hash |
|---|---|---|---|
| `data/dev/` | 100 | selection | `3bb06a95f516cc14` |
| `data/benchmark/` | 75 | selection | `4e42f42c718ba411` |
| `data/validation/` | 81 | held-out test, read once | `5211f72ab632c14b` |

| Artefact | Hash file |
|---|---|
| `data/benchmark/benchmark.csv` | `data/benchmark/benchmark_hash.txt` |
| `data/ground_truth/ground_truth.csv` | `data/ground_truth/ground_truth_hash.txt` |

Benchmark version `nz-dd-v2`. See `BENCHMARK_CONSTRUCTION.md` section 7 for what
changed from v1 and why. Reproducible across Windows / Python 3.13 / pandas
3.0.5 and Linux / Python 3.12 / pandas 3.0.2: identical counts and identical
file hashes.
Python 3.12 / pandas 3.0.2: identical counts and identical file hashes.

**Access rule.** `MWLRDataCHR...csv` holds a coordinate for every description in
the corpus. It is evaluation-only. Only `georef.data.benchmark_builder` may read
it, and `tests/isolation/` fails the build if any other module references it.

## 2. LINZ geographic snapshot

**Source.** NZ Place Names (NZGB), layer `51681` on the LINZ Data Service, the
New Zealand Gazetteer place-names layer maintained by the New Zealand Geographic
Board. Retrieved over WFS 2.0 as JSON.

**Why a snapshot rather than live calls.** LINZ is fetched exactly once. Every
later stage — embedding, ChromaDB, grounding, disambiguation, both experimental
conditions — reads the frozen file. The live service is never called during
prediction. A LINZ update mid-project therefore cannot change results, and
"grounded against LINZ" becomes a claim anyone can check against a hash rather
than an assertion.

**Live schema confirmation.** `--describe` was run against the live service on
2026-09-08. The layer returns 44 property fields. Three corrections followed and
are reflected in `configs/linz.yaml`:

* `land_district`, `region` and `territorial_authority` are three distinct
  fields, not synonyms. `land_district` carries historical survey districts
  (Westland, North Auckland, Nelson), which is the vocabulary most collection
  labels use; `region` holds modern regional-council names and is sparsely
  populated in this layer; `territorial_authority` holds district/city councils.
  All three are retained for regional compatibility scoring.
* `maori_name` is a Yes/No flag indicating whether the name is a Māori name. It
  is **not** an alternative name. It is stored as `is_maori_name` so nothing
  downstream mistakes it for a string to match against.
* `geom_type` records whether the coordinate represents a POINT, LINE or
  POLYGON feature. A polygon's representative point deserves less positional
  weight than a true point, so this is retained.

`numberMatched` comes back as `null` from this service, so paging terminates on
a short page rather than a declared total. The full ingest returned 54,728
records across pages with no duplicate identifiers, consistent with the layer's
published size.

**Recorded per snapshot** (`data/linz/manifests/linz_manifest_<version>.json`):
source authority, dataset name, layer id, dataset version, retrieval timestamp,
endpoint with the key redacted, WFS version, SRS, sort field, field mapping,
bounding boxes, record count, itemised rejection counts, snapshot SHA-256, and
the software environment.

**Deterministic paging.** Requests carry `sortBy=name_id`. Without a stable
server-side sort, WFS pages can overlap or skip records, producing a snapshot
that is wrong in a way no later stage can detect.

**Datum.** Coordinates are the gazetteer's own decimal-degree columns
(`crd_latitude`, `crd_longitude`) in NZGD2000, taken from the properties rather
than the GeoJSON geometry so the stored value is the one LINZ publishes. NZGD2000
and WGS84 differ by well under a metre here, far below the benchmark's error
floor, so no transformation is applied.

**Exclusions, counted not hidden.** The gazetteer covers the Ross Dependency and
the continental shelf. Records outside the configured New Zealand boxes are
excluded and reported as `OUTSIDE_BOUNDING_BOXES`; likewise
`MISSING_REQUIRED_FIELD`, `NON_NUMERIC_COORDINATE` and `DUPLICATE_LINZ_ID`. A
run producing zero usable records raises rather than writing an empty snapshot.

**Unofficial names are kept.** Historical collection localities routinely use
names that were never gazetted as official, so excluding them would remove valid
anchors. `name_status` is retained on every record and is available to the
disambiguator as evidence.

## 3. What the snapshot actually contains

Verified from the live 49,771-record snapshot (`scripts/verify_linz.py`).
Four properties bear directly on later design decisions.

**Names are not unique.** 49,771 records carry 41,402 distinct names. Roughly
one name in six is shared by more than one feature, so name matching alone can
never select a candidate; this is the reason the disambiguator exists, and the
reason score-margin abstention is a real requirement rather than a formality.

**Most features are not points.** 26,369 POINT, 16,886 LINE, 6,516 POLYGON.
Streams alone account for 17,037 records and are lines. The stored coordinate
for a 40 km river is a representative point somewhere along it, so an anchor of
that kind carries a positional uncertainty comparable to the offsets being
measured. `geometry_type` is retained so candidate scoring can down-weight
line and polygon anchors relative to point anchors.

**Most names are unofficial.** About 30,800 of 49,771 records carry an
unofficial status, dominated by 26,596 `Unofficial Recorded`. Excluding them
would have removed the majority of the gazetteer, which confirms the decision to
keep them. `name_status` is retained as disambiguation evidence.

**Administrative fields are unevenly populated.** `land_district` 49,373
(99.2%), `territorial_authority` 49,013 (98.5%), `region` 533 (1.1%). Regional
compatibility scoring must therefore use `land_district` as its primary
vocabulary, which is also the vocabulary historical collection labels use
(Buller, Westland, Nelson). `region` is retained but is too sparse to rely on.
Some records carry two districts (`Canterbury, Westland`), so the field must be
split on commas before comparison rather than matched whole.

## 4. Macrons

The gazetteer holds `Ōpōtiki` and `Whangamatā`; the collection labels in the
benchmark corpus write `Opotiki` and `Whangamata`. NFKC preserves macrons, so
the two forms never match.

`georef.normalise.match_key()` adds diacritic folding on top of `normalise()`
for gazetteer lookup. It is a **separate function by design**: `normalise()`
defines the benchmark join key, and changing it would alter the frozen artefacts
and their published hashes. The frozen hashes are unchanged by this addition:

    benchmark.csv     50d2759cc374350ed278b53b24b5e91e1bf498f56cfc57e01ad47ea2777cb95a
    ground_truth.csv  4e42f42c718ba41138eeeeb89286dbe8af7b1364a81130841d2ca09927f89a4d

Folding is lossy: it cannot distinguish two names differing only by a macron.
The unfolded name is retained on every record, and a folded match is evidence
for a candidate rather than proof of identity.

## 5. Secrets

`LINZ_API_KEY` is read from `.env` (git-ignored) or the environment. It is
interpolated into the request path because LDS requires it there, and redacted
from every manifest, log and error message. Tests assert that a built URL
redacts cleanly, that failure messages do not leak it, and that no manifest
contains it.

## 6. Interpretation stage

`gpt-oss-20b`, served by NVIDIA's OpenAI-compatible endpoint, called through the
official OpenAI SDK. **The same model and the same decoding parameters run both
experimental conditions**; a difference there would measure model capability
rather than the effect of the intervention.

Frozen in `configs/models.yaml`: model identifier, temperature 0, top_p 1,
`max_tokens` 16384, seed, and reasoning effort. Recorded per record: the model
requested, the model the endpoint reported serving, the raw response, the
structured response, the attempt count, and the processing time. Requested and
served identifiers are kept separate because a hosted provider can change served
weights without notice.

Hosted inference is not bit-reproducible even at temperature 0. The run is
**auditable** rather than replayable, and the raw responses are what make it so.

### Prompt versions

| Version | Change | Outcome on the dev set |
|---|---|---|
| `extract-v1` | Single clause per description | 10/10 on a pilot; `geographic_context` unreliable |
| `extract-v2` | Multiple clauses; tighter context and generic-anchor rules | 71/100. 29 schema failures: the model copied hedges into `distance_value` (`"ca. 3"`, `"c. 70"`) and answered `"high"` for confidence |
| `extract-v3` | Prompt asks for a bare number and a separate `distance_is_approximate` flag; validator also parses hedges, fractions and ranges | 0 schema failures. 83/100 on NVIDIA; the other 17 lost to dropped connections |
| `extract-v4` | Back-reference resolution stated as a principle | Ignored by the model; the `anchor_entity` field still said "exactly as written" and won |
| `extract-v5` | Same rule as a numbered five-step procedure, every failing case worked through | **Frozen.** 100/100 interpreted |

### Lenient numeric parsing, and its limit

The validator recovers `ca. 3` → 3, `1/2` → 0.5, and `10-12` → 11. These are
unambiguous readings of what the model returned, not guesses about what it
meant, and losing a whole record to a formatting quirk is the worse error. A
range takes the midpoint, which minimises expected error if the true value lies
anywhere in the interval; this is a fixed pre-benchmark policy.

Leniency stops there. `"a few"` and `"several km"` still fail validation,
because recovering those would mean inventing information the description does
not contain. `confidence: "high"` becomes null rather than a fabricated 0.9.

## 7. Evidence labels

Claims about this pipeline are labelled by the evidence behind them:

| Label | What exists today |
|---|---|
| UNIT TEST | client request construction, paging, retries, redaction, normalisation, snapshot and manifest writing, all against a fake transport |
| REAL OFFLINE INTEGRATION TEST | benchmark construction against the actual supplier files |
| LIVE API SMOKE TEST | `scripts/fetch_linz.py --describe` and a full ingest, both run 2026-09-08 |

**Live run, 2026-09-08.** 54,728 records retrieved; 49,771 kept; 4,957 rejected
as `OUTSIDE_BOUNDING_BOXES` (the layer covers the Ross Dependency and undersea
features); zero rejected for missing fields, non-numeric coordinates or
duplicate identifiers.

One request returned HTTP 400 and the identical URL succeeded on the next
attempt. 400 is deliberately **not** retried, because it normally indicates a
malformed request and silently retrying it would mask real parameter errors. If
this recurs, it should be investigated rather than worked around.

The snapshot hash from that run is superseded: the field-mapping corrections
above change the record contents, so the ingest must be re-run and a new hash
recorded before embedding.
