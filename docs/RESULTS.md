# Development Results

Every measurement taken, what each was actually a test of, and what was
concluded.

**Sections 3 to 8 are selection evidence** — dev and benchmark, used to choose
the configuration. **Section 10 is the reported result**, on 81 held-out records
read once. Section 9 covers Condition C, pre-registered separately.

**Last updated:** after the held-out evaluation. Freeze `0f97cf091a529cf6`.

---

## 0. Status — COMPLETE

The held-out set has been read, once. Freeze `0f97cf091a529cf6`, experiment
version v2. Section 10 is the reported result; everything before it is selection
evidence.

There is no unused data left in this corpus. Any further hypothesis is
exploratory in the strict sense.

### The reported result, in one table

81 held-out records, `gemini-3.8-flash`, same model and parameters in every arm.

| | Coverage | Mean km | Median km | <1 km | >50 km |
|---|---|---|---|---|---|
| A — standalone LLM | 100% | 9.157 | 2.933 | 14.8% | 3.7% |
| B — hybrid pipeline | 66.7% | 9.296 | 2.354 | 25.9% | 3.7% |
| C — agreement @ 25 km | 100% | 9.179 | 3.267 | 24.7% | 3.7% |

| Hypothesis | Outcome |
|---|---|
| **H1** hybrid has lower positional error | **Not supported.** Wilcoxon p = 0.524, effect size 0.087, median paired difference 0.023 km |
| **H2** hybrid coverage is not lower | **Rejected.** 66.7% against 100% |
| **H3** hybrid produces fewer errors over 50 km | **Rejected.** 3.7% against 1.8% like-for-like |
| **H4** C beats A on error | **Not supported.** p = 0.966 |
| **H5** C beats B on error | **Untestable.** n = 3 informative pairs |
| **H6** C coverage equals A | **Holds**, by construction |
| **H7** C has fewer >50 km than B | **Untestable.** n = 3 |
| **H8** C precision not below A | **Holds directionally.** 24.7% against 14.8%, p = 0.077 |

**H1 replicates as a null on two independent sets.** That is a stronger negative
than either alone.

## 0a. The benchmark result (now a selection result)

Run on 75 records, both conditions, `gemini-3.8-flash`, freeze
`562bede96e539187`. Section 8 has the detail; this is the summary.

| Hypothesis | Outcome |
|---|---|
| **H1** hybrid has lower positional error | **Not supported.** Wilcoxon p = 0.777, median paired difference 0.054 km, effect size 0.037 |
| **H2** hybrid coverage is not lower | **Rejected.** 80% against 100% |
| *(all three measured on what is now selection data — see section 8)* | |
| **H3** hybrid produces fewer errors over 50 km | **Rejected.** 5.0% against 1.7% on the paired subset |

One reliable positive finding: sub-kilometre accuracy, **31.7% against 15.0%**,
McNemar p = 0.031 on 18 discordant pairs. It is the only statistically reliable
difference in the comparison and it favours the hybrid.

## 1. What is and is not being tested here

The research question compares two whole pipelines on 75 held-out benchmark
records. Nothing in this document does that. What is measured here is narrower
and prior to it:

| Question | Where it is answered |
|---|---|
| Which configuration should Condition B be frozen with? | Sections 3–8, on the selection sets |
| Does the hybrid beat the standalone baseline? | **Section 10.** No |
| Is the difference statistically reliable? | **Section 10.** No — null on two independent sets |

Every number in sections 3–8 is a **within-Condition-B** comparison: one variant
of the hybrid against another. Cross-condition results appear only in sections 9
and 10.

### Why these are separate

Choosing the configuration by measuring it requires ground truth. Doing that on
the benchmark would be tuning on the evaluation set, and the result would be
worthless. The dev set exists so the choice can be measured on records the
benchmark will never see: 100 records, disjoint from the benchmark by
description and by ground-truth coordinate, both disjointness properties
asserted by tests.

The cost of this design is that a configuration fitted to 100 records may not
generalise to 75 different ones. That cost is stated rather than hidden, and
section 6 says where the fit is fragile.

---

## 2. Setup common to every run

| Component | Value |
|---|---|
| Interpretation model | `gemini-3.8-flash`, `reasoning_effort: medium`, temperature 0 |
| Prompt | `extract-v5` |
| Gazetteer | LINZ NZ Place Names, 49,771 records, frozen snapshot |
| Retrieval | exact key, diacritic-folded key, BM25, dense (`all-MiniLM-L6-v2`), fused by RRF |
| Reranking | off |
| Geodesy | GeographicLib, WGS84 |
| Ground truth | 100 dev records, MWLR extract |

Extraction succeeded on 100/100 records at 4.3 s each with 0–1 retries.

**The error floor.** Ground truth is largely NZMS 260 map-sheet derived, giving
an inherent positional error of roughly one to a few kilometres. A median of
about 3 km is therefore close to the best any method could show on this data.
Sub-kilometre accuracy claims are not supportable from this benchmark, and the
floor applies identically to both conditions, so the paired comparison stands.

---

## 3. Interpretation prompt versions

Each version fixed a failure the previous one exposed. This is a record of the
failures, not just the final prompt.

| Version | Change | Result |
|---|---|---|
| `extract-v1` | Single clause per description | 10/10 on a pilot. `geographic_context` unreliable |
| `extract-v2` | Multiple clauses; tighter context and generic-anchor rules | **71/100.** 29 schema failures: hedges copied into `distance_value` (`"ca. 3"`), `"high"` for confidence |
| `extract-v3` | Bare number requested; validator parses hedges, fractions, ranges | **0 schema failures.** 83/100 on NVIDIA; the other 17 lost to dropped connections, not interpretation |
| `extract-v4` | Back-reference rule stated as a principle | Ignored. The `anchor_entity` field still said "exactly as written", which contradicted it and won |
| `extract-v5` | Same rule as a numbered five-step procedure, every failing case worked through | **100/100.** Back-references resolve |

### What v5 fixed

Descriptions name a feature once then refer back to it. v4 asked the model to
resolve this and it did not; v5 walks it through the reasoning step by step.

| Description | v4 anchor | v5 anchor |
|---|---|---|
| `Lindis Pass … 200m west of Pass` | `Pass` | `Lindis Pass` |
| `Mt. Maungatua … 500 m west of the summit` | `the summit` | `Mt. Maungatua` |
| `Perpendicular Point by trig. 1/2 mile N of the point` | `the point` | `Perpendicular Point` |
| `bank of Mangahowhi Stream … 10 m NW from the bridge` | `the bridge across the stream` | `Mangahowhi Stream` |

There is also a deterministic fallback in the grounder, so this does not depend
on the prompt alone.

---

## 4. Ablations run

### 4.1 Two-pass retrieval hints

**Tested:** whether telling the model which names in a description are real New
Zealand gazetteer names improves interpretation. Names only, never coordinates,
so the model still does not know where anything is.

| | Coverage | Median km | Mean km | >50 km |
|---|---|---|---|---|
| Without hints | 74% | 3.15 | 29.39 | 6.8% |
| With hints | 76% | 3.15 | 28.75 | 6.6% |

**Conclusion: marginal, not adopted.** Three records newly scored, one lost to
`UNRESOLVED`, one slightly worse. The drop in `within_1km` from 31.1% to 28.9%
is a denominator effect — three additional mediocre predictions entered the
pool and nothing already scored degraded.

Left implemented and switched off. A component that does not earn its place is
a result, and reporting it costs nothing.

### 4.2 Multi-anchor resolution policy

**Tested:** whether to use the first clause, the shortest offset, or a
disagreement-weighted fusion of both when a description states two relations.

| Policy | Mean km |
|---|---|
| `first` | 29.435 |
| `shortest` | 29.369 |
| `fuse` | 29.385 |

**Conclusion: undecidable on this data.** A 0.2% spread, because only about
three dev records have two clauses. `fuse` is retained as the configured
default on the argument that it abstains when two clauses disagree — but that
is a design argument, not an empirical one, and it must be reported as
untested rather than as a measured choice.

### 4.3 Spatial coherence

**Tested:** whether candidates should be scored on their distance from the
other places the description names. `Nelson, Little Totara River … Costello
Hill` pins a record to the north-west of the South Island regardless of how
well a Southland candidate matches on name.

The headline means initially argued against it. The paired comparison did not.
Of 73 records both configurations scored, **exactly one changed**, and it
improved (10.3 → 4.3 km). Everything else was abstentions converted to
predictions:

| Record | Error | Selected |
|---|---|---|
| DEVGT-067 | 0.2 km | Peria |
| DEVGT-008 | 0.4 km | Akaroa |
| DEVGT-026 | 1.1 km | Mount George |
| DEVGT-011 | 1.7 km | Akaroa |
| DEVGT-021 | 10.6 km | Coal Flat |
| DEVGT-053 | 42.2 km | Salisbury Creek |
| DEVGT-046 | 106.1 km | Little River |
| DEVGT-051 | 227.9 km | Mill Arm |

**Conclusion: adopted.** It degrades nothing and adds six good predictions for
two poor ones, and the two poor ones are separable by selection score.

Judging it on the mean alone would have rejected a component that improves
everything it touches. That is worth stating as a methodological point: on a
distribution this skewed, the mean measures the tail, not the method.

### 4.4 Abstention threshold

**Tested:** where selection confidence stops predicting reliability.

| Coherence | `minimum_score` | Coverage | Mean km | Median km | >50 km |
|---|---|---|---|---|---|
| off | 0.35 | 74 | 15.94 | 3.20 | 3 |
| off | 0.40 | 70 | 8.59 | 3.10 | 2 |
| off | 0.45 | 66 | 8.84 | 3.10 | 2 |
| on | 0.35 | 81 | 19.25 | 3.10 | 5 |
| on | 0.40 | 80 | 19.36 | 3.10 | 5 |
| **on** | **0.45** | **74** | **9.46** | **3.07** | **3** |
| on | 0.50 | 72 | 9.55 | 3.07 | 3 |

**Conclusion: 0.45.** Same coverage as the previous configuration with the mean
cut from 15.94 to 9.46 km. Records excluded between 0.40 and 0.45 average
around 140 km of error.

Coherence-off at 0.40 is leaner on mean but reaches four fewer records, and
H2 concerns coverage.

---

## 5. Configuration frozen for the benchmark

| Parameter | Value | Basis |
|---|---|---|
| Prompt | `extract-v5` | 100/100 interpreted |
| Retrieval channels | exact, folded, BM25, dense | Not individually ablated |
| Reranking | off | Not measured; cost not justified without evidence |
| Two-pass hints | off | Measured, marginal |
| Spatial coherence | on | Measured, positive |
| `minimum_score` | 0.45 | Measured, section 4.4 |
| Multi-anchor policy | `fuse` | **Untested**, design argument only |

---

## 6. Where this is fragile

**Two parameters are fitted to 100 records.** `minimum_score` and the coherence
switch were chosen by minimising dev-set error. The cliff between 0.40 and 0.45
rests on a handful of records. The full sensitivity table is published above so
the fragility is visible rather than buried in a single chosen number.

**The mean is dominated by its tail.** Median 3.07 km against mean 9.46 km. Any
comparison reported on means alone will be a comparison of a few outliers.
Median, the threshold proportions and the >50 km rate carry more information
here, and the write-up should lead with them.

**The multi-anchor policy is unmeasured.** Three multi-clause records cannot
distinguish three policies. If the benchmark contains meaningfully more, the
policy remains an untested component of Condition B and must be described that
way.

**Coverage is not accuracy.** Condition B abstains where an anchor is not in a
place-names gazetteer — roads, huts, structures, unnamed features. Condition A
will emit coordinates for all of them. That difference is what H2 measures, and
comparing accuracy across different denominators without saying so would be
misleading. Every table above states its denominator.

---

## 7. Bugs found by measurement, not by testing

Recorded because each was invisible until a number looked wrong, and each
changed a result.

| Symptom | Cause |
|---|---|
| 29 schema failures | Prompt said "copy as written"; the model copied `"ca. 3"` into a float field |
| 17 records lost | 8 workers exceeded the endpoint's capacity; 45 of 83 successes needed retries |
| Coverage stuck at 48% | Abstaining between `Akaroa` the Locality and `Akaroa` the Town, which are the same place |
| A 203 km error | A **Hill** named Pukenui beat the **Locality**, because a bare anchor gave feature type no weight |
| Coherence made things worse | `North Island` matched scattered features and became a context point |
| `Little River` 502 km wrong | LINZ gazettes `Wairewa/Lake Forsyth`; a bare `Lake Forsyth` lookup missed it |
| Mean fell 29 → 16 for no reason | Adding a feature weight rescaled every score and moved two records under the abstention margin. An accidental improvement, not an evidential one |
| `--no-coherence` had no effect | The flag was wired but the spotter was never attached. Both runs were the same configuration |

The last two are the reason the run header now prints what is **actually**
attached rather than what the configuration requests, and why a feature that
scores its neutral value on every record raises a warning.


---

## 8. The benchmark run — NOW A SELECTION RESULT

> **Protocol revision.** After this run was evaluated, the benchmark was
> promoted from held-out test to selection data, and the previously unused
> 81-record set became the held-out test. See
> `CONDITION_C_PREREGISTRATION.md` section 2a for the reasoning and what it
> costs. **The result below is a development result, not a headline finding.**
> It is reported because it is what motivated the revision and because a
> disclosed history is part of the record.

Freeze `562bede96e539187`. Condition A run `20260909T170347-2863c560`,
Condition B run `20260909T170602-e3106dae`, evaluation `20260909T171343`.
Both conditions ran `gemini-3.8-flash` at `reasoning_effort: medium`, on the
same 75 descriptions, in the same session.

### Coverage

| | Records answered | Coverage |
|---|---|---|
| Condition A | 75 | 100% |
| Condition B | 60 | 80% |

Condition B abstained 15 times, all `GROUNDING_FAILURE`. Condition A never
abstained: asked for coordinates, it always produced some.

### Like-for-like, the 60 records both conditions answered

| | Condition A | Condition B |
|---|---|---|
| mean error km | 7.263 | 11.185 |
| **median error km** | **3.373** | **3.375** |
| p90 error km | 13.024 | 20.330 |
| within 1 km | 15.0% | **31.7%** |
| within 5 km | 61.7% | 61.7% |
| over 50 km | 1.7% | 5.0% |

### Paired tests

| Test | Result |
|---|---|
| Wilcoxon signed-rank, paired errors | W = 876, **p = 0.777**, n = 60, median difference 0.054 km, r = 0.037 |
| McNemar, within 1 km | **p = 0.031**, 18 discordant |
| McNemar, within 5 km | p = 1.000, 20 discordant |
| McNemar, within 10 km | p = 0.581, 13 discordant |
| McNemar, within 50 km | p = 0.500, 2 discordant |

### What the numbers say

**The medians are identical.** 3.373 against 3.375 km, on the same records.
Both sit at the benchmark's own error floor, which is roughly one to a few
kilometres of NZMS 260 map-sheet imprecision. Neither method can demonstrate
anything below that, and both reach it.

**The hybrid is better when it is right and worse when it is wrong.** Nearly
twice as many predictions within a kilometre, and three times as many beyond
fifty. Deterministic geodesy from an authoritative coordinate is precise; the
same machinery applied to a wrongly-selected anchor is precisely wrong.

**The abstentions were avoidable.** Condition A scored *better* on the 15
records Condition B declined (mean 4.994, median 2.107, 73.3% within 5 km) than
on the 60 it kept (mean 7.263, median 3.373, 61.7%). The hybrid abstained on
the **easy** cases. Its coverage cost bought nothing.

That is a direct consequence of the abstention threshold. `minimum_score` was
raised from 0.35 to 0.45 on the dev set because it cut dev mean error from 15.94
to 9.46 km at equal coverage. On the benchmark it removed 15 records the
baseline handled comfortably. The fragility flagged in section 6 — two
parameters fitted to 100 records — is exactly what materialised.

**The threshold is not being changed.** Re-tuning it now would be
benchmark-driven tuning, and the result would be worthless. A sensitivity
analysis may be reported as clearly-labelled exploratory analysis, after the
frozen result, and never as the headline.

### Why the baseline is this strong

The most important thing in this result is not the hybrid's performance but the
baseline's. A 2026 frontier model, given only a locality description, places 75
of 75 New Zealand records at a median error of 2.877 km with no retrieval, no
gazetteer and no geodesy.

The plain reading is that a model of this generation has absorbed enough New
Zealand geography that authoritative grounding adds little **on average**. Where
grounding still wins is precision: the model knows roughly where Waiouru is, but
only a geodesic computed from an authoritative coordinate lands within a
kilometre of a point 19 km southwest of it. That is exactly where the hybrid's
one reliable advantage appears.

This is a scope limitation, not a design flaw. The specification asked for a
"small LLM", and the experiment ended up running a capable hosted one because
the small-model endpoint was unusable. Against a genuinely small or older model,
the comparison would very likely look different — and that is a stated,
testable claim rather than a hedge.

### What an examiner will ask, and the answers

**Is this a failed experiment?** No. The design isolates a controlled
intervention and the intervention did not produce the hypothesised effect on
average. A null result from a sound design is a result. The hypotheses in
section 3 of the specification were written to be tested, not confirmed.

**Could the hybrid be made to win?** Almost certainly, by lowering the
abstention threshold — which is precisely why that must not be done now.

**Is n = 60 enough?** For the Wilcoxon, the observed effect size of 0.037 means
no plausible sample size would make this difference practically meaningful. For
McNemar at 1 km, 18 discordant pairs supporting p = 0.031 is thin but real, and
should be reported with that caveat attached.


---

## 9. Condition C — combination rules

### ⚠ POST-BENCHMARK EXPLORATORY ANALYSIS

**Condition C was conceived after the A-versus-B benchmark was evaluated.** A
rule invented to fit an observed result and then reported against that same
result proves nothing. Nothing in this section replaces the frozen comparison in
section 8, and nothing here is a headline finding.

It is held to the same standards as the rest of the work, because the label is a
disclosure, not an exemption:

* every rule is specified from the design rather than from the outcome, and
  each could have been written before any run;
* thresholds are fixed on the **dev set**, using a Condition A dev run, exactly
  as `minimum_score` and the coherence switch were;
* denominators are stated;
* the label is attached to every artefact the analysis writes.

### Why it is worth looking at

Section 8 showed the two conditions failing *differently* rather than one
dominating. Like-for-like on 60 records, the medians are 3.373 and 3.375 km —
indistinguishable — but the hybrid is more than twice as likely to land within a
kilometre and three times as likely to miss by fifty. Two methods with different
failure modes are the standard case where a combination can beat both.

### The rules

| Strategy | Rule | Motivation |
|---|---|---|
| `fallback` | Hybrid where it answered, baseline where it abstained | What a deployed system would do. Recovers full coverage without discarding any hybrid prediction |
| `agreement` | Hybrid when the two estimates agree within *k* km, else baseline | Two methods sharing no machinery arriving at the same place is evidence neither produces alone. **Uses no ground truth and no fitted threshold beyond *k*** |
| `confidence` | Hybrid when its disambiguation score clears a threshold, else baseline | Uses the hybrid's own self-assessment, available at prediction time |
| `prefer_baseline` | Baseline unless it abstained | The control. Any rule that does not beat this achieves nothing |

`agreement` carries the least methodological baggage: it reads only the two
predictions, so it could have been specified before either condition existed.

### Constraints the implementation enforces

No rule can invent a coordinate — every output is one of the two inputs,
unchanged, and a test asserts it. Every record records which condition it came
from and why. Records where neither condition predicted are reported as
`NO_PREDICTION` rather than filled in.

### Running it

```bash
# Fix the thresholds on the dev set first
python scripts/run_benchmark.py --condition A --dataset dev --profile gemini
python scripts/combine_conditions.py --dataset dev \
    --baseline <dev_run> --hybrid <dev_run>

# Then apply the chosen rule to the benchmark
python scripts/combine_conditions.py --baseline <run> --hybrid <run> \
    --strategy agreement --agreement-km <chosen on dev>
```

The script costs nothing: it runs no model and makes no request, reading two
finished prediction files and applying arithmetic.

Running it directly on the benchmark prints a warning, because a threshold
chosen with the benchmark in view cannot be reported as validated. The dev path
exists so the choice is made where choices are allowed to be made.

### What this can and cannot conclude

It **can** show whether the two conditions' errors are complementary, and by how
much — which is a genuine property of the methods and interesting regardless of
which rule wins.

It **cannot** be reported as a third experimental condition of the dissertation,
because it was not pre-registered, and its parameters would have to be fixed on
dev *before* the benchmark for that claim to hold. If a future version of this
work pre-registers a combination rule and freezes it alongside A and B, it
becomes a legitimate third arm. Here it is exploratory analysis, and it is
labelled as such in every file it writes.


---

## 10. The held-out result

Freeze `0f97cf091a529cf6`, version v2. Condition A run `20260909T221535-9f7d5b08`,
Condition B run `20260909T221751-1552d9a1`, combination `20260909T222554`,
evaluation `20260909T222558`. Read once.

### Like-for-like, the 54 records both conditions answered

| | Condition A | Condition B |
|---|---|---|
| mean error km | 6.657 | 9.296 |
| **median error km** | 2.584 | **2.354** |
| p90 error km | **10.967** | 22.784 |
| within 1 km | 13.0% | **25.9%** |
| within 5 km | **68.5%** | 64.8% |
| over 50 km | **1.8%** | 3.7% |

Wilcoxon p = 0.524, effect size 0.087, median paired difference 0.023 km.

The pattern is the same one the benchmark showed, and it is now the settled
characterisation: **the hybrid is better when it is right and worse when it is
wrong.** Twice the sub-kilometre rate, twice the catastrophic rate, double the
p90. The middle of the distribution is unchanged.

### The reversal: abstention was well calibrated here

On the 27 records Condition B declined, Condition A scored **worse** than on the
54 it kept:

| Condition A on | Mean km | Median km | within 5 km |
|---|---|---|---|
| the 54 both answered | 6.657 | 2.584 | 68.5% |
| the 27 B declined | **14.158** | **4.129** | **51.8%** |

**Condition B abstained on the genuinely hard cases.** Its coverage cost bought
accuracy, and the abstention mechanism did what it was designed to do.

This is the **opposite** of what the benchmark showed, where B declined records
the baseline handled better than average. Same pool, same construction rule,
same threshold, opposite behaviour on two samples of 75 and 81.

That contradiction is the most useful thing in the result. It says the
abstention threshold is not merely mis-set but **unstable**: at
`minimum_score = 0.45` the boundary falls in a region where a small change of
sample flips which records land on which side. A parameter fitted to 100 records
and behaving inconsistently across two further samples of comparable size is not
a tuned parameter, it is a coin toss with a decimal point.

Coverage across the three sets — 74%, 80%, 66.7% — carries the same message.

### Condition C adds nothing

`agreement` @ 25 km drew from B on 51 records and A on 30, restoring coverage to
100%. Against Condition A: Wilcoxon p = 0.966. Its median, 3.267 km, is worse
than **both** A (2.933) and B (2.354).

H5 and H7 are untestable: C and B agree on all but 3 of the records both
answered, so there is nothing for a paired test to work with. That is not a
missing analysis, it is what the rule does — where B answered and the two agreed,
C *is* B.

The rule's failure is now explicable. Agreement between the two conditions
concentrates on records where both were already close, so C inherits B's values
on the easy cases and A's on the hard ones — and lands worse in the middle of
the distribution than either. On the selection set this looked like the best of
the four rules; on held-out data it is worse than doing nothing.

**That is exactly what a held-out set is for**, and the selection-versus-held-out
gap here is a cleaner demonstration of overfitting than anything that could have
been arranged deliberately.

### What can be claimed

1. **Retrieval and deterministic grounding do not improve average positional
   accuracy against a 2026 frontier model on this task.** A null on two
   independent held-out sets, with effect sizes of 0.037 and 0.087.
2. **They do change the error profile.** Roughly double the sub-kilometre rate,
   roughly double the catastrophic rate, on both sets.
3. **Abstention costs coverage and its value is sample-dependent.** Well
   calibrated on one set, badly on another, at the same threshold.
4. **A post-hoc combination rule selected on 175 records did not generalise to
   81.** Worse than either component it combines.

### What cannot be claimed

That the hybrid is better, or worse, on average — the evidence supports neither.
That the sub-kilometre advantage is established: p = 0.077 here against p = 0.003
on the selection set, so it is suggestive and unconfirmed. That any of this
generalises to a smaller model, which is the scope limitation in section 8 and
the single most valuable follow-up.

### Why the baseline is this strong

Given only a locality description, with no gazetteer, no retrieval and no
geodesy, `gemini-3.8-flash` placed 81 of 81 New Zealand records at a median
error of 2.933 km — inside the benchmark's own NZMS 260 error floor.

The honest reading is that a model of this generation has absorbed enough New
Zealand geography that authoritative grounding adds little on average. The
specification asked for a small LLM; the small-model endpoint proved unusable
and the experiment ran a capable hosted one instead. Against a genuinely small
or older model the comparison would very likely look different, and that is a
testable claim rather than a hedge.
