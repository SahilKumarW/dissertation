# Condition C — Pre-Registration

**Status:** written before the held-out set has been read.
**Evaluation set:** `data/validation/` — 81 records, never used for anything.
**Written:** after the A/B benchmark was evaluated. See section 2 and section
2a, which are the parts that matter.

---

## 1. What is being pre-registered

A third experimental condition combining the standalone baseline and the hybrid
pipeline, and the hypotheses it will be tested against.

This document is written and committed **before** the validation set is read.
Everything it fixes — the rules, how the operating rule is chosen, the metrics,
the tests, the denominators — is fixed here rather than after the numbers
arrive.

## 2. Why a third set exists, and the honesty problem it solves

Condition C was conceived after seeing that A and B fail *differently* on the
benchmark: like-for-like, medians of 3.373 and 3.375 km, but the hybrid twice as
likely to land within a kilometre and three times as likely to miss by fifty.
That observation motivated the idea.

A rule motivated by an observed result cannot be tested against that same
result. Reporting it against the benchmark would be a post-hoc rule evaluated on
the data that suggested it, which proves nothing however carefully it is
implemented.

The construction in `BENCHMARK_CONSTRUCTION.md` produced 256 independent
eligible records. The benchmark took 75 and the dev set 100. **81 were never
allocated**, by the same deterministic hash ordering, and have never been read,
scored, or inspected.

That set is spent here. It can be spent once: after this, there is no unused
data left, and any further hypothesis on this corpus is exploratory in the
strict sense and must say so.

### What this does and does not buy

It **does** mean the rule is evaluated on data that did not suggest it, which is
the standard remedy and the reason validation sets exist.

It does **not** erase the origin of the idea. The dissertation must state that
Condition C was designed after observing the benchmark and evaluated on a
held-out set, not imply it was conceived alongside A and B. The design is
legitimate; the history is still part of the record.

## 2a. PROTOCOL REVISION — the benchmark is now a selection set

**Declared change, made after the A/B benchmark had been run and evaluated.**

### What changed

| Set | n | Original role | Role from this point |
|---|---|---|---|
| `data/dev/` | 100 | selection | selection |
| `data/benchmark/` | 75 | **held-out test** | **selection** |
| `data/validation/` | 81 | unused | **held-out test** |

Every tunable parameter — `minimum_score`, the coherence switch, the
multi-anchor policy, and Condition C's rule and threshold — is selected on the
pooled **175** records of dev plus benchmark. The reported result for all three
conditions is the 81-record held-out set.

### Why

The benchmark exposed a concrete failure of selection on 100 records:
`minimum_score` was raised from 0.35 to 0.45 because it cut dev mean error from
15.94 to 9.46 km, and on the benchmark it discarded 15 records that the baseline
handled **better** than average. The parameter was fitted to too little data and
it generalised badly.

Pooling dev and benchmark gives 175 records to select on instead of 100, which
is the direct remedy. The 81 records exist precisely so a held-out test survives
the change.

### What this costs, stated plainly

**The A/B benchmark result in `RESULTS.md` section 8 is now a development
result, not a headline finding.** It cannot be reported as a held-out
evaluation, because its data is now used for selection. Reporting both it and
the validation result as if each were a primary outcome would invite the
question of which would have been led with had the numbers differed — and that
question would be fair.

The history stands as part of the record: the benchmark was originally the test
set, it was run, it was evaluated, and the protocol was then revised. A revision
disclosed with its motivation and its date is ordinary practice; the same
revision undisclosed is not.

### What it does not change

The 81 held-out records have never been read, scored or inspected. Nothing about
this revision touches them, and the stopping rule in section 8 applies to them
unchanged: **one pass**.

## 3. The rules

Four deterministic combination rules, implemented in
`georef.pipeline.condition_c`. Each reads two finished prediction records and
applies arithmetic. **No rule may produce a coordinate that is not one of its
two inputs**, and a test enforces this.

| Strategy | Rule |
|---|---|
| `fallback` | Hybrid where it answered, baseline where it abstained |
| `agreement` | Hybrid when the two estimates agree within *k* km, else baseline |
| `confidence` | Hybrid when its disambiguation score ≥ *s*, else baseline |
| `prefer_baseline` | Baseline unless it abstained. **The control** |

`prefer_baseline` is not a candidate. It exists so that any rule which fails to
beat it is recognised as having achieved nothing.

## 4. How the operating rule is chosen

**On the pooled selection set — dev plus benchmark, 175 records — once, before
the held-out set is read.**

Both conditions have now been run on both sets, so the pooled selection can be
made directly:

```bash
python scripts/combine_conditions.py --sweep \
    --pair dev:<dev_A_run>:<dev_B_run> \
    --pair benchmark:<bench_A_run>:<bench_B_run>
```

Selection criterion, fixed here: **the lowest median paired geodesic error
against Condition A on the dev set**, with ties broken toward the rule with the
lower proportion of errors over 50 km, and remaining ties toward `agreement` on
the grounds that it uses no fitted threshold beyond *k*.

*k* and *s* are chosen on the dev set over these grids and no others:

* *k* ∈ {2, 5, 10, 25, 50} km
* *s* ∈ {0.45, 0.55, 0.65, 0.75}

The chosen strategy and parameter are written into `configs/condition_c.yaml`,
hashed by the freeze, and **not changed afterwards**.

## 5. Hypotheses

Stated as directional predictions, to be tested, not assumed.

**H4 — accuracy against the baseline.** Condition C will have lower paired
geodesic error than Condition A on the validation set.
*Test:* Wilcoxon signed-rank, two-sided, paired over records where both produced
a coordinate.

**H5 — accuracy against the hybrid.** Condition C will have lower paired
geodesic error than Condition B on the validation set.
*Test:* Wilcoxon signed-rank, two-sided, paired.

**H6 — coverage.** Condition C will have coverage equal to Condition A.
This is expected to hold **by construction**, since every rule falls back to the
baseline, and is stated so a failure would be visible as a bug rather than a
finding.

**H7 — catastrophic errors.** Condition C will produce a smaller proportion of
errors over 50 km than Condition B.
*Test:* McNemar, exact, on paired threshold outcomes.

**H8 — precision is not sacrificed.** Condition C's proportion of predictions
within 1 km will not be lower than Condition A's.
*Test:* McNemar, exact, at the 1 km threshold.

H4 and H5 are the ones that matter. A rule that beats one condition but not the
other has not combined them; it has picked one.

## 5a. Selection evidence available so far

From the dev set alone (100 records), before pooling and before the grid was
searched:

| | Coverage | Mean km | Median km | <1 km | >50 km |
|---|---|---|---|---|---|
| A (baseline) | 1.000 | 11.229 | 2.859 | 0.230 | 0.030 |
| B (hybrid) | 0.740 | 9.543 | 3.056 | 0.324 | 0.041 |
| C `fallback` | 1.000 | 14.470 | 3.041 | 0.310 | 0.050 |
| C `confidence` | 1.000 | 13.899 | 3.089 | 0.290 | 0.050 |
| **C `agreement`** | **1.000** | **11.119** | **2.736** | **0.310** | **0.030** |
| control `prefer_baseline` | 1.000 | 11.229 | 2.859 | 0.230 | 0.030 |

`agreement` is the only rule that beats the control on median error (2.736
against 2.859) while matching it on catastrophic errors (0.030), and it lifts
sub-kilometre accuracy from 23.0% to 31.0% with McNemar p = 0.039. `fallback`
and `confidence` both *raise* the >50 km rate to 0.050, which is the hybrid's
tail leaking through.

This is consistent with section 9's prediction being wrong in an interesting
way: the errors **are** complementary, but only a rule that requires the two
conditions to corroborate each other captures it. Taking the hybrid whenever it
answered, as `fallback` does, imports its tail along with its precision.

## 5b. THE SELECTION — made, and closed

Pooled selection set: dev 100 + benchmark 75 = **175 records**. Registered grids
searched in full. Sweep output `experiments/combined/20260909T220536`.

| Rule | Median km | >50 km | <1 km | Beats control? |
|---|---|---|---|---|
| **`agreement` @ 25 km** | **2.627** | **0.023** | **0.309** | **yes** |
| `agreement` @ 50 km | 2.737 | 0.023 | 0.309 | yes |
| `agreement` @ 10 km | 2.775 | 0.023 | 0.286 | yes |
| `confidence` @ 0.65 | 2.864 | 0.029 | 0.251 | yes |
| `fallback` | 2.864 | 0.046 | 0.309 | yes |
| `confidence` @ 0.45 | 2.864 | 0.046 | 0.309 | yes |
| `confidence` @ 0.55 | 2.864 | 0.046 | 0.297 | yes |
| control `prefer_baseline` | 2.877 | 0.023 | 0.206 | — |
| `confidence` @ 0.75 | 2.877 | 0.023 | 0.206 | no |
| `agreement` @ 5 km | 2.970 | 0.023 | 0.280 | no |
| `agreement` @ 2 km | 3.039 | 0.023 | 0.257 | no |

**Selected: `agreement`, *k* = 25 km.** Lowest median by the registered
criterion, tying the control on catastrophic errors.

Written into `configs/condition_c.yaml` and hashed by the freeze. **Not changed
again.**

### What the sweep shows about the two conditions

Every `agreement` variant holds >50 km at 0.023, the control's rate, while
`fallback` and `confidence` at low thresholds double it to 0.046. Both of the
latter take the hybrid's prediction whenever it exists or is nominally
confident, which imports its tail along with its precision. Requiring the two
independent estimates to **corroborate each other** is what separates them.

The threshold behaves sensibly rather than arbitrarily: tighter agreement
(2 km, 5 km) is too strict and defers to the baseline so often that it loses the
hybrid's precision, while 50 km is loose enough to start admitting
disagreements. 25 km sits between, and it is worth noting that this is close to
the 25 km clause-disagreement threshold chosen independently for the multi-anchor
policy — the scale at which two estimates of one locality stop corroborating
appears to be a property of the data rather than of either rule.

### An honest expectation, recorded before the held-out run

Across every variant, the Wilcoxon against Condition A is far from significant
(p = 0.56 for the selected rule; no variant below 0.21) while McNemar at 1 km is
strongly significant (p = 0.0029). The median paired difference is 0.0 km
everywhere.

The plain reading: **Condition C's benefit is precision, not overall error.** On
the held-out set, H4 and H5 as written are more likely to fail than to pass,
while H8 should pass comfortably. That is stated here, before the data is read,
so the outcome cannot be re-narrated afterwards.

If that is how it lands, the finding is that combining the conditions buys
sub-kilometre accuracy at no cost in catastrophic error or coverage, while
leaving the bulk of the error distribution unchanged — which is a precise and
defensible claim, and a more interesting one than a blanket improvement.

## 6. Metrics and denominators

Primary metric unchanged: WGS84 ellipsoidal geodesic error in kilometres, lower
being better positional accuracy.

Reported for all three conditions on the validation set: coverage, mean, median,
standard deviation, min, max, 25th/75th/90th/95th percentiles, proportions
within 1/5/10 km, proportion over 50 km, and abstention counts by reason.

Every paired test runs over records where **both** compared conditions produced
a coordinate, and that denominator is reported next to the result. Summary
statistics computed over different denominators are not compared.

Given what the benchmark showed, the **median and the threshold proportions are
the primary reading**; the mean is dominated by a handful of tail records and
will be reported but not led with.

## 7. Procedure, in order

1. Run Condition A over the dev descriptions.
2. Select strategy and parameter on the dev set by the criterion in section 4.
3. Write the choice into `configs/condition_c.yaml`.
4. Freeze (`scripts/freeze.py`), producing a new experiment version.
5. Run Conditions A and B over the **validation** set.
6. Derive Condition C from those two runs under the frozen rule.
7. Evaluate all three and test H4–H8.

Steps 5 and 6 are the first time the validation ground truth is read.

## 8. Stopping rule

**One pass.** The validation set is read once, at step 7, and the result is
reported whatever it says.

If H4 or H5 fails, that is the finding. The rule is not adjusted and re-run: a
second pass over the same data would make the first meaningless, and there is no
third set.

## 9. What a null result would mean

If Condition C does not beat both conditions, the honest conclusion is that the
two conditions' errors are not complementary in a way a simple selection rule
can exploit — which is itself informative, and consistent with the benchmark
finding that their medians are identical.

That would suggest the hybrid's advantage at fine precision and its
disadvantage in the tail arise on the *same* records rather than different ones,
so no rule choosing between whole predictions can capture the first without the
second.

## 10. Replication, obtained for free

Running A and B on the validation set also replicates the benchmark comparison
on 81 unseen records with a frozen configuration.

This is a secondary outcome and is reported as such, but it is worth having: a
result that replicates on held-out data is considerably stronger than one that
does not, and a benchmark finding that fails to replicate would be important to
know.
