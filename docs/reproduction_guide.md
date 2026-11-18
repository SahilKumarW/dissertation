# Step-by-Step Reproduction Guide

This guide describes how to reproduce the dissertation experiment and evaluation from a fresh Python environment.

---

## 1. System Requirements and Prerequisites

- **Operating System:** Windows 10/11, Ubuntu Linux 22.04+, or macOS 13+
- **Python Version:** Python `3.11`, `3.12`, or `3.13` (64-bit)
- **Disk Space:** $\ge 5\text{ GB}$ (for PyTorch, Sentence-Transformers, ChromaDB vector store, and LINZ gazetteer)
- **Hardware:** Standard CPU (CUDA GPU optional; all tests and indexing run comfortably on CPU)

---

## 2. Environment Setup

### 2.1 Clone and Navigate to Repository
```bash
git clone <repository_url> dissertation-georef
cd dissertation-georef
```

### 2.2 Create and Activate a Clean Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If execution of scripts is blocked on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2.3 Install Dependencies

Using `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Or using `pyproject.toml` extras:
```bash
pip install -e ".[all]"
```

---

## 3. Configuration and API Keys

Copy `.env.example` to `.env`:

```bash
cp .env.example .env     # Linux / macOS
copy .env.example .env   # Windows
```

Edit `.env` to supply the necessary API keys:

```ini
# Google AI Studio API key (Required for live Gemini 3.8 Flash execution)
# https://aistudio.google.com/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# LINZ Data Service API key (Only needed if re-fetching the live gazetteer from WFS)
# https://data.linz.govt.nz/my/api/
LINZ_API_KEY=your_linz_api_key_here

# NVIDIA NIM API key (Only needed if using the nvidia_gpt_oss profile)
NVIDIA_API_KEY=your_nvidia_api_key_here

# Google Maps API key (Optional / secondary validation)
GOOGLE_MAPS_API_KEY=
```

> [!NOTE]
> All model parameters, temperatures, base URLs, and retry policies are committed and frozen under `configs/`. Secrets alone live in `.env`.

---

## 4. Dataset and Gazetteer Verification

The repository includes pre-built, verified, and hashed datasets:

1. **Held-out Evaluation Dataset (81 records):**
   - Descriptions: `data/validation/validation.csv`
   - Ground Truth: `data/validation/validation_ground_truth.csv`
2. **Benchmark Selection Dataset (75 records):**
   - Descriptions: `data/benchmark/benchmark.csv`
   - Ground Truth: `data/ground_truth/ground_truth.csv`
3. **Development Tuning Dataset (100 records):**
   - Descriptions: `data/dev/dev_descriptions.csv`
   - Ground Truth: `data/dev/dev_ground_truth.csv`
4. **Frozen LINZ Gazetteer Snapshot (49,771 entities):**
   - Normalized JSONL: `data/linz/normalised/linz_place_names_20260908.jsonl`
   - SHA-256 Digest: `data/linz/normalised/linz_place_names_20260908.sha256`

To verify the integrity of the frozen LINZ snapshot:
```bash
python scripts/verify_linz.py
```
*Expected Output:* Reports 49,771 entities, bounding boxes, feature types, and gazetteer anchor coverage.

---

## 5. Building or Verifying the Vector Index

If `data/chroma/` database files are absent, build the dense vector index:
```bash
python scripts/build_index.py
```
- Downloads `sentence-transformers/all-MiniLM-L6-v2`.
- Embeds all 49,771 gazetteer records into ChromaDB collection `nz_geographic_entities`.
- Emits `data/chroma/index_manifest.json` linking the vector index to the snapshot hash.
- *Runtime:* ~2–4 minutes on standard CPU.

---

## 6. Verifying Experiment Freeze State

Before running experiments, verify that the configuration and code match the frozen state `0f97cf091a529cf6`:

```bash
python scripts/freeze.py --verify
```
*Expected Output:* Confirms zero configuration or code drift against `experiments/freeze_manifest.json`.

---

## 7. Re-Running the Evaluation on Existing Predictions

The repository preserves the exact frozen prediction runs reported in the dissertation under `experiments/`. You can reproduce the full statistical evaluation instantly without making live API calls:

```bash
python scripts/evaluate.py \
    --baseline 20260909T221535-9f7d5b08 \
    --hybrid 20260909T221751-1552d9a1 \
    --dataset validation
```

This generates an evaluation folder under `experiments/evaluation/<timestamp>/` containing:
- `metrics.json`: Per-condition summary metrics (coverage, mean, median, percentiles, threshold proportions).
- `statistics.json`: Paired Wilcoxon signed-rank and McNemar test statistics.
- `paired_results.csv`: Case-by-case paired comparison table.

### 7.1 Running Condition C Combination Rule
To reproduce Condition C arbitration (`agreement` within 25 km):
```bash
python scripts/combine_conditions.py \
    --baseline 20260909T221535-9f7d5b08 \
    --hybrid 20260909T221751-1552d9a1 \
    --dataset validation \
    --strategy agreement \
    --agreement-km 25
```

---

## 8. Executing Live Experiments (Requires Gemini API Key)

To execute fresh live inference across the 81 held-out validation descriptions:

### 8.1 Test Endpoint Connectivity
```bash
python scripts/check_llm.py --profile gemini
```

### 8.2 Run Condition A (Standalone Baseline)
```bash
python scripts/run_benchmark.py --condition A --dataset validation --profile gemini
```
Outputs predictions to `experiments/validation_baseline/<run_id>/predictions.jsonl`.

### 8.3 Run Condition B (Hybrid Pipeline)
```bash
python scripts/run_benchmark.py --condition B --dataset validation --profile gemini
```
Outputs predictions to `experiments/validation_hybrid/<run_id>/predictions.jsonl`.

### 8.4 Evaluate the New Runs
```bash
python scripts/evaluate.py --baseline <new_baseline_run_id> --hybrid <new_hybrid_run_id> --dataset validation
```

---

## 9. Running Automated Test Suite

Run the full automated test suite:

```bash
# Run unit and isolation tests (fast, offline)
pytest -q -m "not slow"

# Run all tests including slow corpus rebuild validation
pytest -q
```

---

## 10. Status of Steps: Tested vs. Documented

| Step / Workflow | Verification Status | Notes |
|---|---|---|
| Dependency Installation | **VERIFIED** | Verified with clean pip resolution on Python 3.12. |
| Test Suite Execution | **VERIFIED** | Unit and isolation tests pass offline with zero API calls. |
| Offline Evaluation Script | **VERIFIED** | `scripts/evaluate.py` successfully reproduces reported metrics. |
| Condition C Combination | **VERIFIED** | `scripts/combine_conditions.py` reproduces reported agreement results. |
| LINZ Snapshot Ingestion | **DOCUMENTED** | Snapshot is committed and hashed; re-fetch from LINZ WFS requires active `LINZ_API_KEY`. |
| Live Gemini API Execution | **DOCUMENTED** | Live calls require external `GEMINI_API_KEY` and consume Google AI Studio API quota. |
