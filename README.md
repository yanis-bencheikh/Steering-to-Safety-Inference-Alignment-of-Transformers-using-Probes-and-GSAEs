# 🛡️ Steering to Safety: Inference Alignment of Transformers Using Probes and GSAEs

> **Mila – Quebec AI Institute | McGill University | HEC Montréal**  
> Yanis Bencheikh, Aoudou Njingouo Mounchingam, Fabrice Leroy Tiojip Latche, Eva Portelance, Danilo Bzdok, Mina Arzaghi

---

## 📌 Overview

Large Language Models (LLMs) often exhibit unsafe behaviors at inference time. This project benchmarks two complementary **inference-time steering** methods — supervised **Linear Probes** and unsupervised **Gated Sparse Autoencoders (GSAEs)** — applied to a frozen **RoBERTa** backbone, without any model retraining.

We show that:
- **Linear Probes** are highly effective at reducing overall toxicity and harmfulness
- **GSAEs** discover 51 interpretable "atoms" correlated with harmful concepts
- **Combining both** (probe + relevant SAE atoms) yields synergistic gains in jailbreak compliance rates

---

## 🧠 Architecture

### Backbone
- **Model:** `roberta-base` (d = 768), fully **frozen** during all experiments
- **Pooling:** Mean pooling over token activations → single sentence vector H ∈ ℝ⁷⁶⁸

### Gated Sparse Autoencoder (GSAE)
Implements the **DeepMind Gated SAE** architecture (Rajamanoharan et al., 2024):

```
f(x) = π(x) ⊙ r(x)
```

- **π(x)** — Gating path (sparsity): `𝕀(W_enc^T x + b_gate > 0)` ∈ {0,1}^k
- **r(x)** — Magnitude path: `ReLU(W_enc^T x + b_mag)` ∈ ℝ^k
- **Expansion factor:** 64× → **k = 49,152 latent features**
- **Optimizer:** Constrained Adam (unit-norm decoder columns)
- **Loss:** L_total = L_reconstruct + λ · L_sparsity + L_aux

> The Gated architecture solves the **shrinkage bias** of vanilla SAEs by decoupling gate (which neurons fire) from magnitude (how strongly), preventing the optimizer from artificially suppressing signal amplitudes.

### Linear Probes
- **Architecture:** Logistic Regression on frozen RoBERTa activations H
- **Steering vector:** `v = probe.coef_[0]` (normalized), used as activation direction
- **Intervention:** `h' = h ± λ · v` (Activation Engineering at inference time)

---

## 📦 Datasets

| Dataset | Size | Role |
|---|---|---|
| **BeaverTails** (PKU-Alignment) | 300k+ QA pairs | Harmfulness probe training (14 harm categories) |
| **CivilComments** | 1.8M comments | Toxicity probe training (24 identity metadata cols) |
| **GoEmotions** | 58k Reddit comments | Emotional atom discovery (27 fine-grained labels) |
| **EmpatheticDialogues** | 25k conversations | Empathy/gratitude steering synergy |
| **CrowS-Pairs** | 1,508 pairs | OOD bias evaluation (masked LM) |
| **StereoSet** | 2,106 examples | OOD stereotype evaluation |
| **Wikipedia** | ~2M articles | GSAE pretraining corpus (generic language) |

**Data loading strategy:** A robust ETL pipeline with Google Drive caching ("download once, cache forever") handles all datasets — including a custom `tarfile`-based loader for EmpatheticDialogues due to an unstable HF script.

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/steering-to-safety.git
cd steering-to-safety

# Install dependencies
pip install transformers datasets huggingface_hub pandas tqdm torch scikit-learn
```

### Hugging Face Authentication
Some datasets (BeaverTails, CivilComments) require HF access:
```bash
export HF_TOKEN=your_token_here
# or set it as a Colab secret
```

### Google Drive (for Colab)
The notebook mounts Drive at `/content/drive/MyDrive/AA_hec_2025/projet_final` for persistent caching of model weights and extracted activations.

---

## 🚀 Pipeline

### 1. Data Loading & ETL
```python
# Loads and standardizes all 7 datasets, merges validation splits,
# and caches to Google Drive.
# → Outputs: DatasetDict with canonical 'train'/'test' splits
```

### 2. Activation Extraction (Sharded)
```python
# Passes all datasets through frozen RoBERTa → extracts H vectors (768-dim)
# Sharded in batches of 20,000 with local SSD buffering before Drive transfer
# → Outputs: .npy shards of shape (N, 768)
```

### 3. GSAE Training
```python
# Grid search over:
#   K_EXPANSION ∈ [32, 64]    (24k or 49k latent atoms)
#   L1_LAMBDA   ∈ [1e-4, 5e-5] (sparsity strength)
# Mixed training corpus: Wikipedia + CivilComments + BeaverTails + EmpatheticDialogues
# Checkpointing + early stopping (patience=3)
# → Outputs: best_model.pt
```

### 4. Latent Activation Generation
```python
# Transforms H → Z using trained GSAE
# Stored as float16 for 2× storage efficiency
# → Outputs: .npy shards of shape (N, 49152)
```

### 5. Linear Probe Training
```python
# Trains LogisticRegression on H for:
#   - Toxicity (CivilComments)
#   - Harmfulness (BeaverTails)
# Extracts and normalizes steering vector v = probe.coef_[0] / ||probe.coef_[0]||
```

### 6. Atom Discovery & Mapping
```python
# Streaming one-pass algorithm (Welford-style) to compute per-atom statistics:
#   - Point-Biserial Correlation with harm labels
#   - Δμ = μ_positive - μ_negative (Effect Size / Steering Strength)
# GPT-4 annotation via Logit Lens for semantic labeling
# → 51 atoms selected after GPT-4 relevance filtering
```

### 7. Steering & Evaluation

| Method | Technique |
|---|---|
| **Linear Probe** | Global activation subtraction: `h' = h - λ · v` |
| **SAE Atoms** | Surgical atom subtraction: `z[atom_idx] -= λ` |
| **Probe + SAE** | Global probe + local atom combination (synergy) |

**Evaluation axes:**
- **Fluency:** Pseudo-Log-Likelihood (PLL) guardrails
- **Efficacy:** Probe probability shifts (ΔP)
- **Safety:** Binary Yes/No jailbreak compliance on BeaverTails
- **Generalization:** OOD bias benchmarks (CrowS-Pairs, StereoSet)

---

## 📊 Key Results

- **51 atoms** discovered and validated via GPT-4 annotation
- **Filtering GSAE atoms** by correlation with harmful categories → significant gains in safe refusal rates
- **Probe alone** provides the most efficient steering
- **Probe + SAE atoms** outperforms either method individually on jailbreak benchmarks
- **Unfiltered atoms** can increase unsafe response probability in custom jailbreak settings — a key safety finding

---

## 🔧 Technical Highlights

- **Memory-mapped shard validation** (`mmap_mode='r'`) — verifies GBs of .npy files in seconds without RAM overflow
- **Streaming statistics** — one-pass accumulator algorithm for correlation over millions of examples
- **Float16 compression** for Z vectors — 2× storage reduction with negligible precision loss
- **Industrial checkpointing** — resume training from any epoch, auto-parses hyperparams from filename via regex
- **Compute-Local, Transfer-Later** I/O strategy — avoids Google Drive API rate limits

---

## 📁 Repository Structure

```
.
├── projet_aa_notebook_28_.ipynb   # Main Colab notebook (end-to-end pipeline)
├── README.md
├── assets/
│   └── poster.jpg                 # Conference poster (Mila/McGill/HEC)
└── outputs/                       # Saved models, atom maps, probe vectors
    ├── gsae_best.pt
    ├── probe_toxicity.pkl
    ├── probe_harmfulness.pkl
    └── atom_mapping.csv
```

---

## 📖 References

1. Rajamanoharan et al. (2024). *Improving Dictionary Learning with Gated Sparse Autoencoders.* arXiv:2404.16595
2. Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.*
3. Liu et al. (1979). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692
4. Ji et al. (2024). *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset.* NeurIPS '23
5. Borkan et al. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.* WWW '19
6. Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020
7. Rashkin et al. (2019). *Towards Empathetic Open-domain Dialogue Systems: A New Benchmark and Dataset.* ACL 2019

---

## 🤝 Authors & Affiliations

| Author | Affiliation |
|---|---|
| **Yanis Bencheikh** | HEC Montréal, Mila, McGill University |
| Aoudou Njingouo Mounchingam | HEC Montréal |
| Fabrice Leroy Tiojip Latche | HEC Montréal |
| Eva Portelance | HEC Montréal, Mila |
| Danilo Bzdok | Mila, McGill University |
| Mina Arzaghi | HEC Montréal, Mila, McGill University |

---

## 📜 License

This project is released for academic and research purposes. See `LICENSE` for details.
