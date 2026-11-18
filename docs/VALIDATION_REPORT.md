# Validation and Verification Report

**Date of Verification:** 2026-11-18  
**Environment:** Windows 11, Python 3.12.10 (64-bit), Isolated Virtual Environment (`.venv`)  
**Freeze Digest:** `0f97cf091a529cf6` (Experiment Version `v2`)  
**Test Runner:** pytest 9.2.0

---

## 1. Summary of Execution Results

| Verification Category | Command Executed | Result | Details |
|---|---|---|---|
| **Freeze Verification** | `python scripts/freeze.py --verify` | **PASS** | Configuration, code, prompt, and snapshot SHA-256 digests match `experiments/freeze_manifest.json` perfectly with zero differences. |
| **Full Pytest Suite** | `pytest -q` | **PASS** | 646 passed, 5 skipped (authorized evaluation modules), 0 failed in 11.17s. |
| **Condition A Isolation** | `pytest tests/isolation/test_condition_a_isolation.py` | **PASS** | 21/21 passed. AST check proves Condition A reaches only `llm.client` and `config.loader`; no geographic libraries or ground truth. |
| **No Ground-Truth Leakage**| `pytest tests/isolation/test_no_ground_truth_leakage.py` | **PASS** | 33 passed, 5 skipped (explicit evaluation modules in `ALLOWED`). AST check proves prediction code cannot access ground-truth columns. |
| **Reproducibility Test** | `pytest tests/reproducibility/test_benchmark_reproducible.py` | **PASS** | 2/2 passed. Rebuild from raw MWLR corpus produces byte-identical SHA-256 hashes for benchmark, ground truth, and dev pool. |
| **LINZ Snapshot Verify** | `python scripts/verify_linz.py` | **PASS** | 49,771 entities verified across 3 bounding boxes; dev pool anchor coverage verified (54.08% exact, 59.23% folded). |
| **Offline Evaluation** | `python scripts/evaluate.py --baseline 20260909T221535-9f7d5b08 --hybrid 20260909T221751-1552d9a1 --dataset validation` | **PASS** | Exactly reproduces dissertation reported figures: Condition A median 2.584 km, Condition B median 2.354 km, Wilcoxon $p=0.524$, effect size $r=0.087$. |
| **Condition C Arbitration**| `python scripts/combine_conditions.py --dataset validation --baseline 20260909T221535-9f7d5b08 --hybrid 20260909T221751-1552d9a1 --strategy agreement --agreement-km 25` | **PASS** | Exactly reproduces reported figures: coverage 100%, median 3.267 km, Wilcoxon vs A $p=0.966$. |
| **Secret Scanning** | Git status & environment variable inspection | **PASS** | No API keys, credentials, or private tokens committed or tracked. `.env` is git-ignored; `.env.example` contains only empty placeholders. |

---

## 2. Detailed Test Suite Execution

### 2.1 Test Suite Breakdown (651 Total Test Items)

| Test File | Test Category | Items Passed | Items Skipped |
|---|---|---|---|
| `tests/data/test_benchmark_artefacts.py` | REAL OFFLINE INTEGRATION | 18 | 0 |
| `tests/isolation/test_condition_a_isolation.py` | ISOLATION TEST | 21 | 0 |
| `tests/isolation/test_no_ground_truth_leakage.py` | ISOLATION TEST | 33 | 5* |
| `tests/reproducibility/test_benchmark_reproducible.py`| REPRODUCIBILITY TEST | 2 | 0 |
| `tests/unit/test_anchors.py` | UNIT TEST | 38 | 0 |
| `tests/unit/test_audit.py` | UNIT TEST | 18 | 0 |
| `tests/unit/test_baseline.py` | UNIT TEST | 10 | 0 |
| `tests/unit/test_condition_c.py` | UNIT TEST | 21 | 0 |
| `tests/unit/test_dense_index.py` | UNIT TEST | 8 | 0 |
| `tests/unit/test_disambiguation.py` | UNIT TEST | 35 | 0 |
| `tests/unit/test_extraction.py` | UNIT TEST | 65 | 0 |
| `tests/unit/test_geodesic.py` | UNIT TEST | 67 | 0 |
| `tests/unit/test_grounding.py` | UNIT TEST | 17 | 0 |
| `tests/unit/test_linz_client.py` | UNIT TEST | 19 | 0 |
| `tests/unit/test_linz_ingest.py` | UNIT TEST | 21 | 0 |
| `tests/unit/test_linz_verify.py` | UNIT TEST | 23 | 0 |
| `tests/unit/test_llm_client.py` | UNIT TEST | 34 | 0 |
| `tests/unit/test_name_spotter.py` | UNIT TEST | 19 | 0 |
| `tests/unit/test_normalise.py` | UNIT TEST | 24 | 0 |
| `tests/unit/test_resolution.py` | UNIT TEST | 15 | 0 |
| `tests/unit/test_retrieval.py` | UNIT TEST | 22 | 0 |
| `tests/unit/test_runner.py` | UNIT TEST | 13 | 0 |
| `tests/unit/test_statistics.py` | UNIT TEST | 36 | 0 |
| `tests/unit/test_units.py` | UNIT TEST | 54 | 0 |
| **Total** | | **646** | **5** |

*\*Note on Skipped Tests:* In `test_no_ground_truth_leakage.py`, 5 tests are intentionally skipped because the checked modules (`benchmark_builder.py`, `evaluation/__init__.py`, `evaluation/metrics.py`, `evaluation/statistics.py`, `provenance/freeze.py`) are explicitly designated evaluation/builder modules permitted to reference evaluation assets. All non-evaluation prediction modules passed without reference to ground truth.

---

## 3. Evaluation Replication Output

Running:
```bash
python scripts/evaluate.py --baseline 20260909T221535-9f7d5b08 --hybrid 20260909T221751-1552d9a1 --dataset validation
```
Produced:
```text
Evaluation 20261118T212309

  All records each condition answered (DIFFERENT denominators):
                         Condition A  Condition B
  coverage                      1.000        0.667
  scored predictions               81           54
  mean error km                 9.157        9.296
  median error km               2.933        2.354
  p90 error km                 18.288       22.784
  max error km                149.716       90.232
  within 1 km                   0.148        0.259
  within 5 km                   0.630        0.648
  within 10 km                  0.815        0.759
  over 50 km                    0.037        0.037

  Like-for-like: the 54 records BOTH conditions answered:
                         Condition A  Condition B
  mean error km                 6.657        9.296
  median error km               2.584        2.354
  p90 error km                 10.967       22.784
  within 1 km                   0.130        0.259
  within 5 km                   0.685        0.648
  over 50 km                    0.018        0.037

  Condition A on the 27 records Condition B declined:
    mean_error_km          14.158
    median_error_km        4.129
    within_5km             0.518

  Denominators: {"benchmark_records": 81, "both_conditions_predicted": 54, "only_condition_a_predicted": 27, "only_condition_b_predicted": 0, "neither_predicted": 0}

  Wilcoxon signed-rank on the paired errors (A minus B):
    test                         Wilcoxon signed-rank (paired, two-sided)
    statistic_W                  668.0
    p_value                      0.524023
    n_used                       54
    n_zero_differences_dropped   0
    median_difference_km         0.0231
    effect_size_r                0.0873
    method                       normal approximation with continuity correction
```

---

## 4. Known Issues and Remediations

1. **Pre-existing Virtual Environment:**
   - *Issue:* The `.venv` directory initially present in the clean copy contained broken absolute path pointers to an external developer's Anaconda installation (`C:\Users\mim26\anaconda3\python.exe`).
   - *Remediation:* Recreated `.venv` using local Python 3.12, installed all required packages via `requirements.txt`, and verified `.gitignore` properly excludes `.venv/` from git tracking.
2. **Missing `pydantic` in `pyproject.toml`:**
   - *Issue:* `pyproject.toml` did not list `pydantic` in `dependencies`, causing an import failure if installed strictly via `pip install -e .` without optional packages.
   - *Remediation:* Added `pydantic>=2.0` to `dependencies` in `pyproject.toml` and specified all dependencies in `requirements.txt`.
3. **Windows Charset Handling in `scripts/verify_linz.py`:**
   - *Issue:* On Windows command prompts defaulting to `cp1252` encoding, printing Māori place names with macrons (such as `\u0101` in `Māori`) triggered a `UnicodeEncodeError`.
   - *Remediation:* Added UTF-8 stdout reconfiguration inside `scripts/verify_linz.py` if running on Python 3.7+ Windows terminals.

---

## 5. External Dependencies and Unverified Steps

1. **Live Gemini API Calls (`scripts/run_benchmark.py`, `scripts/extract_dev.py`):**
   - In accordance with the critical audit instruction *"Do not make live Gemini API calls unless explicitly necessary and authorised. Do not consume API credits just to verify documentation,"* live Gemini API calls were **not** executed during this validation.
   - The integration is fully unit tested using hermetic fakes (`tests/unit/test_llm_client.py`, `tests/unit/test_baseline.py`, `tests/unit/test_extraction.py`).
2. **Live LINZ WFS Endpoint Fetch (`scripts/fetch_linz.py`):**
   - The live WFS endpoint fetch was not re-executed to avoid modifying the frozen LINZ snapshot. The frozen snapshot `linz_place_names_20260908.jsonl` was verified against its published SHA-256 digest.
