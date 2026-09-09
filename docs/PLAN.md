# Project Plan

MSc Artificial Intelligence dissertation — natural-language spatial
georeferencing.

**Status: complete.** The held-out set has been read once and the result is
reported in `RESULTS.md`. This file records what was built, in what order, and
the decisions taken along the way.

**Last updated:** after the held-out evaluation, freeze `0f97cf091a529cf6`.

---

## The experiment

Does combining LLM-based spatial interpretation with retrieval, authoritative
geographic grounding and deterministic GIS computation improve the accuracy and
reliability of georeferencing natural-language spatial descriptions containing
explicit distance and directional relationships, compared with standalone LLM
prompting?

Three conditions process identical descriptions with the identical model:

| | Pipeline |
|---|---|
| **A** | description → LLM → coordinates |
| **B** | description → LLM interpretation → retrieval → disambiguation → LINZ anchor → WGS84 geodesy → coordinates, or abstention |
| **C** | a deterministic combination of A and B, pre-registered separately |

The same model and decoding parameters run every arm. A difference there would
measure model capability rather than the intervention.

**Answer: no.** H1 is not supported on two independent held-out sets. Full
result and interpretation in `RESULTS.md`.

---

## Data sets and their roles

| Set | n | Role |
|---|---|---|
| `data/dev/` | 100 | selection |
| `data/benchmark/` | 75 | selection (originally the held-out test) |
| `data/validation/` | 81 | **held-out test — read once** |

The benchmark was promoted from test to selection data partway through, as a
declared protocol revision documented with its motivation and date in
`CONDITION_C_PREREGISTRATION.md` section 2a. The reason: `minimum_score` had
been fitted to 100 records and generalised badly, discarding 15 benchmark
records the baseline handled better than average. Pooling gave 175 records to
select on.

The cost, stated in that document and again here: **the A/B benchmark result is
a development result, not a headline finding.**

---

## What was built

| Step | Component | State |
|---|---|---|
| 1 | Benchmark, ground truth, dev and validation sets, constructed and hashed | Done (`nz-dd-v2`) |
| 2 | LINZ ingestion, frozen snapshot | Done — 49,771 records |
| 3 | ChromaDB index and four-channel hybrid retrieval | Done |
| 4 | Semantic interpretation | Done (`extract-v5`) |
| 5 | Deterministic geodesy and multi-anchor resolution | Done |
| 6 | Candidate disambiguation and abstention | Done |
| 7 | Condition B end to end | Done |
| 8 | Condition A, isolation-tested | Done |
| 9 | Selection on the pooled selection set | Done and closed |
| 10 | Freeze | Done — `0f97cf091a529cf6`, v2 |
| 11 | Held-out run, 81 × A and 81 × B | Done |
| 12 | Evaluation and paired statistics | Done |
| 13 | Condition C, pre-registered and evaluated | Done |
| — | Application (FastAPI + React) | **Not built.** Deliberately deprioritised: it demonstrates the research rather than being it |

646 tests, 5 skipped.

---

## Frozen configuration

| Parameter | Value | Basis |
|---|---|---|
| Model | `gemini-3.8-flash`, temperature 0, `reasoning_effort: medium` | Endpoint availability; see limitations |
| Extraction prompt | `extract-v5` | 100/100 interpreted |
| Retrieval channels | exact, folded, BM25, dense | Not individually ablated |
| Reranking | off | Not measured; cost unjustified without evidence |
| Two-pass hints | off | Measured, marginal |
| Spatial coherence | on | Measured, positive on paired evidence |
| `minimum_score` | 0.45 | Measured; **later shown unstable** |
| Multi-anchor policy | `fuse` | Untested — too few multi-clause records |
| Condition C rule | `agreement` @ 25 km | Selected on 175 records; **did not generalise** |

The last two rows are the honest ones. Both are recorded as such in `RESULTS.md`
rather than presented as validated choices.

---

## Decisions worth keeping

**The model never computes.** Bearings come from a lookup table and distances
from a conversion table. The prompt asks for `"southwest"` and `3.5 miles`,
never `225` and never kilometres. Any bearing the model volunteers is recorded
and ignored, with the disagreement reported.

**Retrieval proposes, disambiguation disposes.** 41,402 distinct names across
49,771 records means one name in six is shared, so every channel returns a list.

**Abstention is a first-class outcome.** Nine named statuses, never collapsed.

**The LLM gets no geographic tools.** Tool calling is used as a serialisation
contract — one schema-shaped function, forced — not as agency. Letting the model
decide when to search would make the intervention unattributable and
non-deterministic. A test asserts the exposed tool set contains nothing
resembling search, geocode, gazetteer or geodesic.

**Endpoint profiles are a development convenience, not an experimental
variable.** `--profile` switches between NVIDIA and Gemini for development; the
frozen runs use one profile across every arm, and every record carries its
provider, profile and served model so a mixed file is detectable.

---

## Things that went wrong, and what they cost

Each was invisible until a number looked wrong. Full list in `RESULTS.md`
section 7; these are the ones that changed a conclusion.

| Problem | Consequence |
|---|---|
| Prompt said "copy as written"; model copied `"ca. 3"` into a float field | 29 of 100 records lost |
| 8 workers exceeded the endpoint's capacity | 17 records lost; 45 of 83 successes needed retries |
| Abstaining between `Akaroa` the Locality and `Akaroa` the Town | Coverage stuck at 48% |
| A **Hill** named Pukenui beat the **Locality** | 203 km error from a bare anchor |
| `North Island` used as a spatial context point | Coherence made results worse than no coherence |
| LINZ dual names (`Wairewa/Lake Forsyth`) unmatched | `Little River` 502 km wrong |
| A feature weight rescaled every score | Mean fell 29 → 16 km by accident, not by evidence |
| `--no-coherence` wired but the spotter never attached | Two runs compared the same configuration |
| Freeze manifest keys used `str(Path)` | Every source file read as changed across platforms |
| `condition_c.yaml` absent from the hashed config list | The selected rule was not covered by the freeze |

The last four were caught by verification rather than by testing, which is the
argument for having built the verification.

---

## Limitations

**The abstention threshold is unstable.** `minimum_score = 0.45` produced
coverage of 74%, 80% and 66.7% on three samples from the same pool, and B
abstained on the *easy* records in one set and the *hard* ones in another. The
threshold sits where a small change of sample flips which records fall on which
side.

**Condition C did not generalise.** Best of four rules on 175 selection
records; worse than both components on 81 held-out ones.

**The multi-anchor policy is untested.** Three policies, a 0.2% spread, and only
about three multi-clause records to distinguish them.

**The model is not small.** The specification asked for a small LLM. The
small-model endpoint proved unusable under load and the experiment ran a capable
hosted model instead. This is the single most consequential limitation: a 2026
frontier model has absorbed enough New Zealand geography that authoritative
grounding adds little on average, and the comparison would very likely look
different against a smaller or older model.

**Ground truth has an error floor.** Largely NZMS 260 map-sheet derived, giving
one to a few kilometres of inherent imprecision. Both conditions reach it; no
sub-kilometre claim is supportable from this benchmark.

---

## Follow-up work, in order of value

1. **Repeat against a genuinely small model.** The infrastructure is built and
   it is a one-line profile change. This is the experiment the specification
   originally described and the one whose answer is least predictable.
2. **Fix the abstention threshold properly**, with the instability treated as
   the phenomenon to study rather than a parameter to re-fit.
3. **Pre-register a combination rule alongside A and B** rather than after them,
   which would make Condition C a legitimate third arm.
4. **Google secondary validation**, specified but never built.
5. **The application layer**, lowest priority.

---

## Document map

| Document | Contents |
|---|---|
| `README.md` | How to install and run every stage |
| `RESULTS.md` | Every measurement, what it tested, what was concluded |
| `ARCHITECTURE.md` | Directory structure, module-by-module reference |
| `BENCHMARK_CONSTRUCTION.md` | Pre-registration of the data sets |
| `CONDITION_C_PREREGISTRATION.md` | Condition C, its hypotheses, the protocol revision |
| `DATA_PROVENANCE.md` | Where each dataset came from and what may read it |
