# Natural-Language Spatial Georeferencing: Standalone LLM vs. Neuro-Symbolic Hybrid Pipeline

**MSc Artificial Intelligence Dissertation — Cardiff University**  
**Author:** Sahil -  
**Status:** Complete & Frozen (`0f97cf091a529cf6`, Experiment Version `v2`)  
**Core Finding:** The neuro-symbolic hybrid pipeline alters the error distribution (substantially increasing sub-kilometre precision but also catastrophic outliers) while failing to demonstrate a statistically significant reduction in median paired geodesic error over the frontier standalone LLM ($p = 0.524$, effect size $r = 0.087$).

---

## 1. Project Overview

Natural-language locality descriptions in biological collection records (e.g. *"Stream gully crossing Highway 8 about 200m west of Lindis Pass"*) are foundational for biodiversity research, species distribution modelling, and conservation planning. Historically, georeferencing these descriptions has relied either on manual cartographic interpretation or rigid rule-based parsing.

This dissertation investigates whether combining the natural language interpretation strengths of a frontier Large Language Model (**Gemini 3.8 Flash**) with deterministic geographic knowledge retrieval (**Land Information New Zealand [LINZ] Gazetteer**) and exact ellipsoidal geodesy (**GeographicLib / Karney WGS84**) improves georeferencing accuracy and reliability compared to a standalone LLM predicting coordinates directly.

---

## 2. Research Objective and Core Hypotheses

### Primary Objective
To design, implement, and rigorously benchmark a neuro-symbolic hybrid georeferencing architecture against a standalone frontier LLM baseline across historical New Zealand botanical collection records.

### Experimental Hypotheses Tested on Held-Out Data (81 Records)
- **H1 (Positional Accuracy):** The hybrid pipeline achieves lower median WGS84 geodesic error than the standalone baseline. *(Outcome: **Not supported**, $p = 0.524$, $r = 0.087$)*.
- **H2 (Coverage):** The hybrid pipeline coverage is not lower than the baseline. *(Outcome: **Rejected**, 66.7% vs. 100.0%)*.
- **H3 (Catastrophic Error Reduction):** The hybrid pipeline produces fewer errors exceeding 50 km. *(Outcome: **Rejected**, 3.7% vs. 1.8% like-for-like)*.
- **Sub-kilometre Precision:** The hybrid pipeline doubles the proportion of predictions landing within 1.0 km of ground truth (25.9% vs. 13.0%, McNemar $p = 0.077$ on held-out validation; $p = 0.031$ on selection benchmark).

---

## 3. Experimental Conditions

### Condition A: Standalone LLM Baseline
- **Model:** `gemini-3.8-flash`, greedy decoding ($\text{temperature} = 0.0$, $\text{top\_p} = 1.0$, `reasoning_effort: medium`).
- **Prompt:** `baseline-v1` ([`src/georef/llm/baseline.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/baseline.py)).
- **Mechanism:** The model reads the raw locality text and outputs its best estimate of latitude, longitude, and confidence via a single forced tool call (`record_coordinates`).
- **Isolation Guarantee:** Zero external geographic data, zero gazetteers, and zero spatial calculations. Grounded purely in parametric memory.

### Condition B: Neuro-Symbolic Hybrid Pipeline
- **Architectural Principle:** *"The language model interprets; deterministic Python computes."*
- **1. Semantic Interpretation:** `gemini-3.8-flash` processes the description with `extract-v5` ([`src/georef/llm/extraction.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/extraction.py)) through forced tool `record_spatial_interpretation`. It identifies named places, extracts offset clauses, resolves linguistic back-references (e.g. *"the summit"* $\to$ *"Mt. Maungatua"*), and identifies broader administrative context without predicting coordinates.
- **2. Multi-Channel Retrieval:** Queries the offline LINZ New Zealand Gazetteer (49,771 entities) across four complementary channels:
  - *Exact Name Match:* Case- and punctuation-normalized hash index.
  - *Folded Name Match:* Macron- and diacritic-insensitive index (e.g. *Ōpōtiki* $\leftrightarrow$ *Opotiki*).
  - *Okapi BM25:* Full-text lexical search over structured gazetteer document templates using `bm25s`.
  - *Dense Semantic Search:* Cosine similarity over 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings in ChromaDB.
- **3. Reciprocal Rank Fusion (RRF):** Fuses ranked channel lists parameter-free ($k=60$) with deterministic tie-breaking on ascending `linz_id`.
- **4. Deterministic Disambiguation:** Scores candidates using a weighted multi-feature sum (name match, token overlap, feature type prior, regional context agreement, and spatial coherence with other mentioned places). Enforces strict abstention if top candidate score $< 0.45$ or score margin $< 0.02$.
- **5. Ellipsoidal Geodesy:** Computes destination coordinates from the grounded anchor entity, bearing, and distance using Karney's direct geodesic algorithm on the WGS84 ellipsoid via GeographicLib.
- **6. Multi-Anchor Resolution & Sanity Checks:** Resolves multi-clause descriptions via inverse-offset weighting (`fuse`), abstains if clauses disagree by $> 25\text{ km}$, and enforces containment within New Zealand bounding boxes.

### Condition C: Deterministic Arbitration (Exploratory / Protocol Revision)
- **Strategy:** `agreement` within 25 km ([`src/georef/pipeline/condition_c.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/pipeline/condition_c.py)).
- **Mechanism:** Selects Condition B when both conditions produce coordinates within 25 km; falls back to Condition A on disagreement or hybrid abstention.

---

## 4. Key Results Summary (Held-Out Evaluation, 81 Records)

Evaluated on 81 held-out historical records read once under freeze `0f97cf091a529cf6`:

| Metric | Condition A (Baseline) | Condition B (Hybrid) | Condition C (Agreement @ 25 km) |
|---|---|---|---|
| **Coverage** | **100.0%** (81/81) | 66.7% (54/81) | **100.0%** (81/81) |
| **Abstentions** | 0 | 27 | 0 |
| **All Predictions Mean Error** | 9.157 km | 9.296 km | 9.179 km |
| **All Predictions Median Error** | 2.933 km | **2.354 km** | 3.267 km |
| **Paired-Valid Mean Error ($n=54$)**| **6.657 km** | 9.296 km | — |
| **Paired-Valid Median Error ($n=54$)**| 2.584 km | **2.354 km** | — |
| **Paired Wilcoxon $p$-value** | \multicolumn{2}{c|}{$p = 0.524$ (two-sided, $W = 668.0$, effect size $r = 0.087$)} | $p = 0.966$ vs A |
| **Sub-kilometre Proportion ($\le 1\text{ km}$)**| 13.0% (7/54) | **25.9%** (14/54) | 24.7% (20/81) |
| **Catastrophic Outliers ($> 50\text{ km}$)** | **1.8%** (1/54) | 3.7% (2/54) | 3.7% (3/81) |

*Core Insight:* The hybrid pipeline is **bimodal**—it doubles sub-kilometre precision when grounding succeeds, but introduces catastrophic errors when disambiguation selects the wrong distant entity. Frontier LLMs possess substantial parametric geographical knowledge of New Zealand, establishing a highly competitive baseline.

---

## 5. System Architecture

```
                    INPUT LOCALITY DESCRIPTION
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   CONDITION A (STANDALONE)               CONDITION B (HYBRID)
   Gemini 3.8 Flash (LLM)                 Gemini 3.8 Flash (LLM)
             |                                     |
      record_coordinates             record_spatial_interpretation
             |                                     |
     Predicted Lat / Lon                   Spatial Clauses
             |                                     |
             |                            Multi-Channel Retrieval
             |                            Exact | Folded | BM25 | Dense
             |                                     |
             |                            Reciprocal Rank Fusion (RRF)
             |                                     |
             |                            Deterministic Disambiguator
             |                            (Abstains if score < 0.45)
             |                                     |
             |                            LINZ Authoritative Anchor
             |                                     |
             |                            WGS84 Geodesy (GeographicLib)
             |                                     |
             |                            Multi-Anchor Clause Resolution
             |                                     |
             |                            NZ Bounding Box Sanity Check
             |                                     |
             +------------------+------------------+
                                |
                                v
                       FINAL PREDICTIONS
                                |
                                v
                     CONDITION C ARBITRATION
                                |
                                v
                       EVALUATION & METRICS
                   (Held-Out Ground Truth CSV)
```

For complete architectural details, see [`docs/architecture.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/architecture.md).

---

## 6. Repository Layout

```text
dissertation-georef/
├── README.md                      # Comprehensive academic project guide
├── requirements.txt               # Pinned Python package dependencies
├── pyproject.toml                 # Package definition and pytest configuration
├── .gitignore                     # Git exclusion rules for environments, caches, secrets
├── .env.example                   # Template for required environment variables
│
├── configs/                       # Pre-registered experimental parameters (Committed & Hashed)
│   ├── benchmark.yaml             # Dataset construction, regex filter, bounding boxes
│   ├── apis.yaml                  # API base URLs, timeouts, concurrency, retry policies
│   ├── models.yaml                # LLM profiles, decoding parameters, embedding model
│   ├── retrieval.yaml             # Channel configurations, RRF k=60, top-k allocations
│   ├── disambiguation.yaml        # Candidate scoring weights, abstention score/margin
│   ├── pipeline.yaml              # Multi-anchor policies, disagreement thresholds
│   ├── condition_c.yaml           # Combination rules and selection criteria
│   └── paths.yaml                 # Filepaths for datasets and ground truth
│
├── src/georef/                    # Core library implementation
│   ├── config/loader.py           # Configuration and profile loader
│   ├── data/benchmark_builder.py  # Dataset sampling and quality gate builder
│   ├── disambiguation/            # Anchor variant generation and candidate scoring
│   ├── evaluation/                # WGS84 geodesic error, Wilcoxon, McNemar statistics
│   ├── linz/                      # LINZ Data Service WFS client, ingestion, validation
│   ├── llm/                       # Gemini client, baseline, and spatial extraction
│   ├── models/spatial.py          # Pydantic schemas for spatial clauses and derivations
│   ├── pipeline/                  # Condition B pipeline, Grounder, Condition C, runner
│   ├── provenance/                # File hashing, freeze manifest creation, drift check
│   ├── retrieval/                 # Exact, folded, BM25, dense retrieval, RRF fusion
│   └── spatial/                   # GeographicLib WGS84 geodesy, units, bounding boxes
│
├── scripts/                       # Stage-by-stage command-line entry points
│   ├── build_benchmark.py         # Build benchmark, dev, and validation datasets
│   ├── fetch_linz.py              # Download fresh WFS snapshot from LINZ Data Service
│   ├── verify_linz.py             # Validate snapshot integrity and dev anchor coverage
│   ├── build_index.py             # Embed LINZ snapshot into ChromaDB with MiniLM-L6-v2
│   ├── check_llm.py               # Live API smoke test for Gemini or NVIDIA profiles
│   ├── extract_dev.py             # Run batch semantic interpretation over dev set
│   ├── ground_dev.py              # Offline grounding and scoring on dev interpretations
│   ├── freeze.py                  # Verify or build experiment SHA-256 freeze manifest
│   ├── run_benchmark.py           # Execute experimental conditions across datasets
│   ├── evaluate.py                # Post-prediction evaluation against ground truth
│   └── combine_conditions.py      # Derive Condition C combination predictions
│
├── data/                          # Research datasets and indexes
│   ├── benchmark/                 # 75-record selection benchmark (no coordinates)
│   ├── ground_truth/              # Benchmark ground-truth coordinates & audit report
│   ├── dev/                       # 100-record tuning set (descriptions & coordinates)
│   ├── validation/                # 81-record held-out test set (read once)
│   ├── linz/normalised/           # Hashed LINZ snapshot (49,771 entities, 25.2 MB)
│   └── chroma/                    # Persistent vector store and index manifest
│
├── experiments/                   # Raw prediction runs, manifests, and evaluation summaries
│   ├── baseline/                  # Condition A benchmark predictions
│   ├── hybrid/                    # Condition B benchmark predictions
│   ├── validation_baseline/       # Condition A held-out predictions
│   ├── validation_hybrid/         # Condition B held-out predictions
│   ├── combined/                  # Condition C arbitration predictions
│   ├── evaluation/                # Official metrics.json, statistics.json, paired CSVs
│   └── freeze_manifest.json       # Master SHA-256 digest of code, configs, datasets
│
├── tests/                         # Automated test suite (651 tests)
│   ├── unit/                      # 19 unit test modules using fakes and mocks
│   ├── isolation/                 # AST tests enforcing modular and data boundaries
│   ├── data/                      # Schema, hash, and disjointness validation tests
│   └── reproducibility/           # Rebuild byte-identity reproducibility tests
│
└── docs/                          # Comprehensive research documentation
    ├── REPOSITORY_AUDIT.md        # Comprehensive component audit and file map
    ├── LLM_IMPLEMENTATION.md      # Direct line references for prompts and API calls
    ├── architecture.md            # Detailed execution flow and Mermaid diagrams
    ├── reproduction_guide.md      # Fresh-environment reproduction guide
    ├── VALIDATION_REPORT.md       # Complete test execution logs and verification
    ├── RESULTS.md                 # Detailed experimental results and statistical analysis
    ├── BENCHMARK_CONSTRUCTION.md  # Dataset sampling and quality gate specification
    └── CONDITION_C_PREREGISTRATION.md # Protocol revision for deterministic arbitration
```

---

## 7. Installation and Dependencies

### Prerequisites
- Python `3.11`, `3.12`, or `3.13` (64-bit)
- Standard operating systems: Windows 10/11, macOS, Linux

### Setup Instructions

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 8. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env    # Linux/macOS
copy .env.example .env  # Windows
```

Populate the required credentials in `.env`:
```ini
# Google AI Studio API key (Required for live Gemini 3.8 Flash execution)
GEMINI_API_KEY=your_gemini_api_key_here

# LINZ Data Service API key (Only needed if fetching a new LINZ snapshot)
LINZ_API_KEY=your_linz_api_key_here

# NVIDIA NIM API key (Only needed if using the nvidia_gpt_oss profile)
NVIDIA_API_KEY=your_nvidia_api_key_here
```

---

## 9. How to Reproduce and Evaluate

### Step 1: Verify Freeze State
Confirm that no code or configuration drift has occurred:
```bash
python scripts/freeze.py --verify
```
*Expected:* `UNCHANGED since freeze 0f97cf091a529cf6`.

### Step 2: Reproduce the Official Evaluation (Instant, Zero API Cost)
Evaluate the frozen prediction runs on the 81 held-out validation descriptions:
```bash
python scripts/evaluate.py \
    --baseline 20260909T221535-9f7d5b08 \
    --hybrid 20260909T221751-1552d9a1 \
    --dataset validation
```
*Expected Output:*
- Condition A median error: **2.584 km**
- Condition B median error: **2.354 km**
- Paired Wilcoxon test: $p = 0.524$, effect size $r = 0.087$
- Output written to `experiments/evaluation/<timestamp>/`.

### Step 3: Reproduce Condition C Arbitration
```bash
python scripts/combine_conditions.py \
    --baseline 20260909T221535-9f7d5b08 \
    --hybrid 20260909T221751-1552d9a1 \
    --dataset validation \
    --strategy agreement \
    --agreement-km 25
```
*Expected Output:* Restores coverage to 100% (81/81), median error 3.267 km.

### Step 4: Run the Automated Test Suite
```bash
# Fast unit and isolation tests
pytest -q -m "not slow"

# Complete test suite including rebuild verification
pytest -q
```
*Expected:* `646 passed, 5 skipped in ~11s`.

### Step 5: (Optional) Running Live Experiments with Gemini API
```bash
# 1. Smoke test endpoint connectivity
python scripts/check_llm.py --profile gemini

# 2. Run Condition A baseline on held-out descriptions
python scripts/run_benchmark.py --condition A --dataset validation --profile gemini

# 3. Run Condition B hybrid pipeline on held-out descriptions
python scripts/run_benchmark.py --condition B --dataset validation --profile gemini
```

---

## 10. Important Source-Code Locations for Academic Review

| Research Component | Exact Source File | Key Functions / Classes |
|---|---|---|
| **Gemini Client & Retries** | [`src/georef/llm/client.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/client.py) | `LlmClient.complete()`, `complete_structured()` |
| **Condition A Prompt & Baseline** | [`src/georef/llm/baseline.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/baseline.py) | `BASELINE_PROMPT`, `BaselineGeoreferencer` |
| **Condition B Prompt & Extractor** | [`src/georef/llm/extraction.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/extraction.py) | `SYSTEM_PROMPT` (`extract-v5`), `SpatialExtractor` |
| **Pydantic Clause Schemas** | [`src/georef/models/spatial.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/models/spatial.py) | `SpatialClause`, `ModelInterpretation`, `derive()` |
| **Lexical & BM25 Retrieval** | [`src/georef/retrieval/lexical.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/retrieval/lexical.py) | `LexicalIndex.exact_name()`, `folded_name()`, `bm25()` |
| **Dense Vector Retrieval** | [`src/georef/retrieval/dense.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/retrieval/dense.py) | `ChromaVectorStore.query()`, `build_index()` |
| **Reciprocal Rank Fusion (RRF)** | [`src/georef/retrieval/candidates.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/retrieval/candidates.py) | `reciprocal_rank_fusion()` ($k=60$) |
| **Entity Disambiguation** | [`src/georef/disambiguation/resolver.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/disambiguation/resolver.py) | `Disambiguator.select()`, `score_candidate()` |
| **WGS84 Ellipsoidal Geodesy** | [`src/georef/spatial/geodesic.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/spatial/geodesic.py) | `destination()`, `distance_between()`, `interpolate()` |
| **Multi-Clause Resolution** | [`src/georef/spatial/resolution.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/spatial/resolution.py) | `resolve_clauses()` (policy `fuse`) |
| **Statistical Testing** | [`src/georef/evaluation/statistics.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/evaluation/statistics.py) | `wilcoxon_signed_rank()`, `mcnemar()`, `pair_conditions()` |

---

## 11. Reproducibility and Experimental Integrity Safeguards

1. **Ground-Truth Isolation:** Mechanically validated by `tests/isolation/test_no_ground_truth_leakage.py`. No prediction-side module can import or reference ground-truth data or coordinates.
2. **Condition A Independence:** Mechanically validated by `tests/isolation/test_condition_a_isolation.py`. The baseline is strictly prohibited from accessing gazetteers, retrieval channels, or geodesic libraries.
3. **Byte-Stable Rebuilds:** Hashed and validated by `tests/reproducibility/test_benchmark_reproducible.py`.
4. **Order-Stable Output Artefacts:** Prediction outputs from multithreaded workers are sorted deterministically by `benchmark_id` before saving, ensuring repeatable SHA-256 digests.
5. **No Tuning on Held-Out Data:** All thresholds (disambiguation score 0.45, margin 0.02, disagreement 25 km, combination rule) were pre-registered and fixed on selection data prior to unblinding the 81 held-out validation descriptions.

---

## 12. Primary Documentation Index

- **Auditing & Codebase Verification:** [`docs/REPOSITORY_AUDIT.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/REPOSITORY_AUDIT.md)
- **Detailed LLM Prompts & API Calls:** [`docs/LLM_IMPLEMENTATION.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/LLM_IMPLEMENTATION.md)
- **Architecture & Sequence Flows:** [`docs/architecture.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/architecture.md)
- **Reproduction Guide:** [`docs/reproduction_guide.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/reproduction_guide.md)
- **Validation & Test Execution Report:** [`docs/VALIDATION_REPORT.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/VALIDATION_REPORT.md)
- **Detailed Experimental Findings:** [`docs/RESULTS.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/RESULTS.md)
- **Benchmark Construction Protocol:** [`docs/BENCHMARK_CONSTRUCTION.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/BENCHMARK_CONSTRUCTION.md)
- **Condition C Pre-Registration:** [`docs/CONDITION_C_PREREGISTRATION.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/CONDITION_C_PREREGISTRATION.md)
- **Data Provenance & Licensing:** [`docs/DATA_PROVENANCE.md`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/docs/DATA_PROVENANCE.md)
