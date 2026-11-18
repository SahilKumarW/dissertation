# Repository Audit: Neuro-Symbolic Natural-Language Georeferencing

**Project Title:** Natural-Language Spatial Georeferencing: Standalone LLM vs. Neuro-Symbolic Retrieval Pipeline  
**Degree:** MSc Artificial Intelligence, Cardiff University  
**Audit Date:** 2026-11-18  
**Repository State:** Frozen at `0f97cf091a529cf6` (Experiment Version `v2`)

---

## 1. Executive Summary

This repository implements a controlled experimental comparison between:
- **Condition A (Standalone LLM Baseline):** Direct coordinate estimation using Gemini 3.8 Flash (`gemini-3.8-flash`) through structured tool calling (`record_coordinates`), with zero external geographic data, retrieval, or spatial calculation.
- **Condition B (Neuro-Symbolic Hybrid Pipeline):** Two-stage pipeline combining LLM interpretation (semantic extraction into spatial clauses via `record_spatial_interpretation`) with deterministic geographic retrieval from the LINZ New Zealand Gazetteer (exact key, diacritic-folded, Okapi BM25, and dense embeddings via `all-MiniLM-L6-v2` fused with Reciprocal Rank Fusion), deterministic candidate scoring/disambiguation, and WGS84 ellipsoidal direct geodesic computation using GeographicLib (Karney).
- **Condition C (Post-hoc Deterministic Combination):** Deterministic arbitration between Condition A and Condition B outputs (e.g. `agreement` within 25 km).

The evaluation is conducted on **81 held-out historical New Zealand biodiversity locality descriptions** from the Manaaki Whenua – Landcare Research (MWLR) herbarium collection.

The reported experimental findings are:
- **Standalone LLM Coverage:** 81/81 (100%)
- **Hybrid Valid Results:** 54/81 (66.7%), with 27 abstentions
- **Paired-Valid Median Geodesic Error:** 2.584 km for standalone LLM vs. 2.354 km for the hybrid pipeline
- **Wilcoxon Signed-Rank Test:** $p = 0.524$, effect size $r = 0.087$ (no statistically significant difference on paired errors)
- **Sub-kilometre Accuracy:** 25.9% for hybrid vs. 13.0% for standalone LLM ($p = 0.077$ on held-out validation)

---

## 2. Component Verification Table

| Component | Actual File Path | Purpose | Verified |
|---|---|---|---|
| **1. Main Entry Points** | `scripts/build_benchmark.py` | Constructs benchmark, dev, and validation datasets from raw MWLR source files. | Yes |
| | `scripts/fetch_linz.py` | Downloads WFS layer 50282 (NZ Place Names) from LINZ Data Service into a normalized snapshot. | Yes |
| | `scripts/verify_linz.py` | Inspects LINZ snapshot integrity, coordinate bounds, and dev pool anchor coverage. | Yes |
| | `scripts/build_index.py` | Embeds LINZ snapshot using `all-MiniLM-L6-v2` into persistent ChromaDB collection. | Yes |
| | `scripts/check_llm.py` | Smoke tests LLM provider endpoints (NVIDIA NIM or Google Gemini) with prompt round-trip. | Yes |
| | `scripts/extract_dev.py` | Runs batch interpretation on dev pool descriptions, checkpointing responses to JSONL. | Yes |
| | `scripts/ground_dev.py` | Offline evaluation of dev interpretations against LINZ and ChromaDB without LLM calls. | Yes |
| | `scripts/freeze.py` | Computes SHA-256 digests of configs, code, datasets, and prompts; detects drift with `--verify`. | Yes |
| | `scripts/run_benchmark.py` | Orchestrates execution of Condition A, Condition B, or both across datasets without ground truth access. | Yes |
| | `scripts/evaluate.py` | Loads ground truth, joins predictions, computes WGS84 error, and runs Wilcoxon/McNemar tests. | Yes |
| | `scripts/combine_conditions.py` | Derives Condition C predictions from finished A and B runs using deterministic combination rules. | Yes |
| **2. Gemini API Integration** | `src/georef/llm/client.py` | `LlmClient` wrapping OpenAI SDK pointed at Gemini OpenAI-compatible base URL. | Yes |
| | `configs/apis.yaml` | Service endpoint configuration: URL, API key environment variable, timeout, retry policy. | Yes |
| | `configs/models.yaml` | Model parameters: `gemini-3.8-flash`, temperature 0.0, top_p 1.0, max_tokens 32768, reasoning_effort `medium`. | Yes |
| **3. Prompts** | `src/georef/llm/baseline.py` | `BASELINE_PROMPT` (`baseline-v1`) requesting single best estimate of coordinates via tool. | Yes |
| | `src/georef/llm/extraction.py` | `SYSTEM_PROMPT` (`extract-v5`) 5-step numbered extraction procedure enforcing back-reference resolution. | Yes |
| **4. Parsing & Validation** | `src/georef/llm/client.py` | `complete_structured()` enforcing single schema-shaped function call as serialization contract. | Yes |
| | `src/georef/llm/baseline.py` | Pydantic validation via `BaselineCoordinates` (bounds checking on latitude, longitude, confidence). | Yes |
| | `src/georef/models/spatial.py` | Pydantic validation via `ModelInterpretation` and `SpatialClause`, plus deterministic derivation in `derive()`. | Yes |
| | `src/georef/normalise.py` | Text normalisation (Unicode NFKC, quote/dash canonicalisation, whitespace, lowercasing). | Yes |
| **5. Standalone Implementation (Condition A)** | `src/georef/llm/baseline.py` | `BaselineGeoreferencer`: direct coordinate extraction from LLM with zero retrieval or spatial tools. | Yes |
| **6. Hybrid Implementation (Condition B)** | `src/georef/pipeline/condition_b.py` | `ConditionBPipeline`: joins `SpatialExtractor` with `Grounder`, capturing full end-to-end provenance. | Yes |
| | `src/georef/pipeline/grounding.py` | `Grounder`: orchestrates retrieval, disambiguation, geodesy, and bounding box checks. | Yes |
| | `src/georef/pipeline/runner.py` | `run_parallel()` and `JsonlCheckpoint`: thread-pool execution with order-stable sorting on completion. | Yes |
| **7. Gazetteer Retrieval** | `src/georef/linz/client.py` | WFS client for LINZ Data Service Layer 50282. | Yes |
| | `src/georef/linz/ingest.py` | LINZ record ingestion, field normalisation, and SHA-256 snapshot manifest creation. | Yes |
| | `src/georef/retrieval/lexical.py` | `LexicalIndex`: in-memory exact matching (`normalise()`) and diacritic-folded matching (`match_key()`). | Yes |
| **8. BM25 & Dense Retrieval** | `src/georef/retrieval/lexical.py` | Okapi BM25 implementation using `bm25s` over structured gazetteer document templates. | Yes |
| | `src/georef/retrieval/dense.py` | Dense vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` against persistent ChromaDB. | Yes |
| | `src/georef/retrieval/rerank.py` | Cross-encoder reranking using `BAAI/bge-reranker-v2-m3` (optional ablation). | Yes |
| **9. Rank Fusion** | `src/georef/retrieval/candidates.py` | `reciprocal_rank_fusion()`: parameter-free RRF ($k=60$) with deterministic tie-breaking. | Yes |
| | `src/georef/retrieval/hybrid.py` | `HybridRetriever`: executes enabled channels, applies RRF, and formats candidate list. | Yes |
| **10. Disambiguation & Resolution** | `src/georef/disambiguation/anchors.py` | Anchor variant generator, qualifier stripper, generic-anchor detector, and token overlap similarity. | Yes |
| | `src/georef/disambiguation/resolver.py` | `Disambiguator`: multi-feature scoring, minimum score filter (0.45), minimum margin filter (0.02). | Yes |
| | `src/georef/retrieval/name_spotter.py` | `NameSpotter`: detects other gazetteer names in locality text for spatial coherence scoring. | Yes |
| **11. Geodesic Calculations** | `src/georef/spatial/geodesic.py` | WGS84 direct problem, inverse problem, and geodesic interpolation via GeographicLib (Karney). | Yes |
| | `src/georef/spatial/resolution.py` | Multi-anchor resolution (`first`, `shortest`, `fuse` inverse-offset weighting) and 25km disagreement check. | Yes |
| | `src/georef/spatial/bounds.py` | Point-in-polygon bounding box containment for mainland NZ, Chatham Islands, and offshore islands. | Yes |
| | `src/georef/spatial/units.py` | Deterministic parsing of 16-point compass directions and distance unit conversions to metres. | Yes |
| **12. Dataset Preprocessing** | `src/georef/data/benchmark_builder.py` | Benchmark builder with pre-registered eligibility regex, quality gates, and deduplication. | Yes |
| **13. Evaluation & Statistics** | `src/georef/evaluation/metrics.py` | Geodesic error calculation, error percentiles, coverage, threshold hit rates. | Yes |
| | `src/georef/evaluation/statistics.py` | Exact paired Wilcoxon signed-rank test and exact McNemar test with explicit denominators. | Yes |
| **14. Test Suite** | `tests/conftest.py` | Shared pytest fixtures and dataset checks. | Yes |
| | `tests/isolation/test_condition_a_isolation.py` | AST inspection verifying Condition A imports no retrieval, gazetteer, or geodesy modules. | Yes |
| | `tests/isolation/test_no_ground_truth_leakage.py` | AST inspection verifying prediction code never accesses ground-truth columns or files. | Yes |
| | `tests/reproducibility/test_benchmark_reproducible.py` | Verifies benchmark builder reproduces exact byte hashes. | Yes |
| | `tests/unit/test_*.py` (19 files) | Comprehensive unit tests for all modules with isolated fakes and mock objects. | Yes |
| **15. Configuration & Secrets** | `configs/*.yaml` (9 files) | Pre-registered experiment parameters: models, apis, paths, benchmark, retrieval, pipeline, disambiguation, condition_c. | Yes |
| | `src/georef/config/loader.py` | Type-safe loader for YAML configurations, path resolution, and profile selection. | Yes |
| | `.env.example` | Placeholder environment variable definitions (`GEMINI_API_KEY`, `NVIDIA_API_KEY`, etc.). | Yes |

---

## 3. Inventory of Non-Source Artefacts and Cleanup Actions

### 3.1 Unnecessary / Generated Artefacts
1. **`.venv/`**: A virtual environment directory copied from an external Windows environment (`C:\Users\mim26\anaconda3\python.exe`) whose Python interpreter path does not exist on this machine. Must be excluded from distributable repository and replaced with a clean local virtual environment for testing.
2. **`__pycache__/` and `*.pyc`**: Python bytecode files present across `src/` and `tests/`. Ignored in `.gitignore`.
3. **`.pytest_cache/`**: Pytest cache directory in root. Ignored in `.gitignore`.
4. **`src/dissertation_georef.egg-info/`**: Build metadata generated during `pip install -e .`. Untracked from clean distribution.

### 3.2 Research Assets Requiring Preservation
1. **`data/benchmark/`**: `benchmark.csv`, `benchmark_hash.txt`, `benchmark_manifest.json` (75 records).
2. **`data/ground_truth/`**: `ground_truth.csv`, `ground_truth_hash.txt`, `audit_report.json`.
3. **`data/dev/`**: `dev_pool.csv`, `dev_ground_truth.csv`, `dev_descriptions.csv`, `interpretations_*.jsonl`, `dev_predictions_*.jsonl`.
4. **`data/validation/`**: `validation.csv`, `validation_ground_truth.csv` (81 held-out test records).
5. **`data/linz/normalised/`**: `linz_place_names_20260908.jsonl` (49,771 gazetteer records, 25.2 MB) and SHA-256 digest.
6. **`data/chroma/index_manifest.json`**: Vector store provenance metadata connecting ChromaDB to the exact LINZ snapshot digest.
7. **`experiments/`**: Raw predictions, run manifests, combined metrics, and evaluation summaries (`experiments/baseline/`, `experiments/hybrid/`, `experiments/combined/`, `experiments/evaluation/`, `experiments/validation_*/`).
8. **`experiments/freeze_manifest.json`**: Machine-specific freeze digest (`0f97cf091a529cf6`).

### 3.3 Missing Documentation and Files Resolved
1. **`requirements.txt`**: Was missing from root. Created with full dependency specifications.
2. **`pyproject.toml` dependencies**: Added `pydantic>=2.0` and optional dependencies (`dev`, `retrieval`, `all`).
3. **`docs/REPOSITORY_AUDIT.md`**: Created (this document).
4. **`docs/LLM_IMPLEMENTATION.md`**: Created to provide transparent inspection of Gemini API calls and prompts.
5. **`docs/architecture.md`**: Created with comprehensive component and data flow diagrams.
6. **`docs/reproduction_guide.md`**: Created with reproducible instructions from a fresh Python environment.
7. **`docs/VALIDATION_REPORT.md`**: Created with actual test execution logs and verification checks.
