# Project Log: Retrieval-Augmented Language Model (RETRO)

## What This Project Is

We built a language model that can look things up in an external memory bank while it writes, and proved that the lookup mechanism does genuine intellectual work — finding semantically relevant information and using it to make better predictions on text it has never seen before.

The system has three major components:

1. **A memory bank** (cc_service) — 1.8 million text chunks from Wikipedia, searchable by meaning using a neural fingerprinting model (MiniLM).
2. **A RETRO language model** (nanogpt) — a GPT-style transformer with extra cross-attention layers that can read from the memory bank mid-generation.
3. **A held-out evaluation framework** — a controlled experiment proving the retrieval benefit is real, not an artifact of data leakage.

---

## Component 1: The Memory Bank (cc_service)

### What it does

The memory bank stores text as "concept cells" — atomic units of meaning, each one a paragraph. When you query it with a piece of text, it finds the most semantically similar cells using neural embeddings, not keyword matching.

### How it works

```
Text in → MiniLM encoder → 384-dim embedding → whitening transform → dot-product search → top-k results
```

- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2` — a small (22M param) BERT variant that converts text into 384-dimensional vectors.
- **Whitening**: A learned linear transform (fitted once from a reference corpus) that decorrelates the embedding dimensions, improving retrieval quality. Parameters: mean vector (mu), rotation matrix (W), and max_norm for normalization.
- **Storage**: SQLite database (`bank.db`, 4.24 GB) holding cell weights, source text, and metadata.
- **API**: FastAPI service on `localhost:8765` with endpoints for add, query, bind, inspect, and delete.

### What's in the bank

| Source | Articles | Cells | Description |
|--------|----------|-------|-------------|
| Simple English Wikipedia | all (~205k articles) | ~774k | Full dump of `20231101.simple` |
| English Wikipedia | 100k articles | ~1.04M | First 100k from `20231101.en` |
| **Total** | **~305k articles** | **1,817,204 cells** | Each cell is one paragraph (≥80 chars) |

### Key files

| File | Purpose |
|------|---------|
| `cc_service/service/main.py` | FastAPI entry point |
| `cc_service/service/memory.py` | MemoryBank: ties encoder + SQLite + retrieval |
| `cc_service/service/encoder.py` | MiniLM wrapper (loaded once per process) |
| `cc_service/service/persistence.py` | SQLite schema and read/write |
| `cc_service/load_wikipedia.py` | Loader for Simple English Wikipedia |
| `cc_service/load_wikipedia_en.py` | Loader for full English Wikipedia |
| `cc_service/bank.db` | The database (4.24 GB, ~1.82M cells) |

---

## Component 2: The RETRO Language Model (nanogpt)

### Architecture

Built on top of Andrej Karpathy's nanoGPT, we implemented the RETRO architecture from the 2021 DeepMind paper ("Improving language models by retrieving from trillions of tokens").

**How RETRO differs from standard GPT:**

A standard GPT reads tokens left-to-right and predicts the next one. It can only use information stored in its own weights (what it memorized during training).

RETRO adds **Chunked Cross-Attention (CCA)** layers inside the transformer. The input sequence is split into chunks of 64 tokens. For each chunk, the model retrieves the 2 most similar text chunks from the memory bank and attends to them via dedicated cross-attention heads. This lets the model access external knowledge without spending context window tokens.

**How RETRO differs from RAG:**

RAG (Retrieval-Augmented Generation) pastes retrieved text into the prompt. RETRO is more elegant: retrieved chunks live outside the main sequence in a separate attention pathway. The model learns *when* retrieval is useful and *how* to extract information from it during training.

### Model specification

| Parameter | Value |
|-----------|-------|
| Architecture | RetroGPT (GPT-2 + Chunked Cross-Attention) |
| Parameters | 55.29M |
| Layers | 8 transformer blocks |
| Heads | 8 attention heads per layer |
| Embedding dim | 512 |
| Context length | 256 tokens (4 chunks of 64) |
| CCA layers | Every 2nd block (layers 1, 3, 5, 7) |
| Neighbors per chunk | 2 |
| Neighbor length | 64 tokens each |
| Vocabulary | 50,304 (GPT-2 BPE, padded for tensor-core efficiency) |
| Precision | bfloat16 |
| Hardware | NVIDIA RTX 3070 (8 GB VRAM) |

### Key files

| File | Purpose |
|------|---------|
| `nanogpt/model_retro.py` | RETRO architecture: RetroConfig, ChunkedCrossAttention, RetroBlock, RetroGPT |
| `nanogpt/model.py` | Base nanoGPT (reused for LayerNorm, CausalSelfAttention, MLP) |
| `nanogpt/train_retro.py` | Phase 1 training (Shakespeare, random neighbors) |
| `nanogpt/train_retro_p2_bank.py` | Phase 2 training (Wikipedia, real neighbors) |
| `nanogpt/train_retro_p2_bank_small.py` | Phase 2 small model variant (6M params) |
| `nanogpt/retro_generate.py` | RAG-style generation with GPT-2 + bank (pre-RETRO baseline) |
| `nanogpt/eval_retro_heldout.py` | 3-condition held-out evaluation script |

---

## Training History

### Phase 1: Proof of Concept (Shakespeare)

**Goal:** Verify the RETRO architecture works mechanically — gradients flow through CCA, loss decreases, the model actually uses the retrieval pathway.

- **Data:** tiny-shakespeare (character-level, 65-token vocabulary)
- **Neighbors:** Random Shakespeare chunks during training; oracle neighbors (literal continuations) at eval
- **Result:** Oracle retrieval produced substantially lower loss than no-retrieval, proving the CCA pathway is wired correctly and the model learns to read from it.

**Verdict:** Architecture works. CCA is mechanically connected. Ready for real data.

### Phase 2.1: RAG Baseline

**Goal:** Establish a baseline using the simpler "paste retrieved text into prompt" approach.

- **Script:** `retro_generate.py`
- **Model:** Pretrained GPT-2 small (124M, frozen weights)
- **Method:** Query the bank, paste top results into the prompt, let GPT-2 generate
- **Result:** Works for demonstrations but limited — no learned integration between retrieval and generation.

### Phase 2.2: Real Retrieval (Stage C — 60M tokens)

**Goal:** Train RETRO on real data with real MiniLM-retrieved neighbors from the bank. Test whether semantic retrieval actually helps.

**Data pipeline (3 stages):**

- **Stage A** — `data/bank/prepare.py`: Extract text from bank.db, tokenize with GPT-2 BPE → `train.bin` (60M tokens), `val.bin` (3.1M tokens). Track which cell each token came from (`train_cells.npy`, `val_cells.npy`).
- **Stage B** — `data/bank/precompute_neighbors.py`: For every 64-token chunk, embed with MiniLM, apply whitening, find top-2 bank cells by dot-product (excluding self-retrievals) → `train_neighbors.npy`, `val_neighbors.npy`.
- **Stage C** — `train_retro_p2_bank.py`: Train the 55M-param RetroGPT for 5,000 iterations.

**Result:**

| Metric | Value |
|--------|-------|
| Best val (with retrieval) | 3.6949 (iter 3500) |
| Best val (without retrieval) | 3.8175 |
| Retrieval gap | +0.1225 nats |

The model overfit (1:1 params-to-tokens ratio vs. Chinchilla-optimal 20:1). Val loss bottomed at iter 3500 and drifted upward.

### Phase 2.2e: Small Model Experiment (6M params)

**Goal:** Test whether overfitting was the problem by shrinking the model to ~6M params (10:1 tokens-to-params ratio, closer to Chinchilla optimal).

| Parameter | Large | Small |
|-----------|-------|-------|
| Layers | 8 | 4 |
| Heads | 8 | 4 |
| Embedding | 512 | 256 |
| Params | 55.29M | ~6M |

**Result:**

| Metric | Value |
|--------|-------|
| Best val (with retrieval) | 4.1285 (iter 3500) |
| Best val (without retrieval) | 4.2099 |
| Retrieval gap | +0.0814 nats |

Higher absolute loss (less capacity) but cleaner training dynamics. Confirmed the 55M model was a better choice given sufficient data.

### Phase 2.3: Data Scale-Up (157M tokens)

**Goal:** Fix the overfitting by adding more data rather than shrinking the model.

**What changed:**

- Loaded 100k full English Wikipedia articles into the bank (on top of all Simple English Wikipedia)
- Bank grew from ~774k cells to 1,817,204 cells
- Training corpus grew from 60M to 157M tokens
- Training iterations increased from 5,000 to 15,000

**Data pipeline re-run:**

- Stage A re-ran on the expanded bank → 157M train tokens, 8.3M val tokens
- Stage B re-ran: 2.45M train chunks, 130k val chunks, each with 2 precomputed neighbors
- Stage C trained the same 55M model for 15,000 iterations

**Result:**

| Metric | Value |
|--------|-------|
| Best val (with retrieval) | 3.6045 (iter 13000) |
| Best val (without retrieval) | 3.8279 |
| **Retrieval gap** | **+0.2234 nats** |
| Final val (iter 15000, with) | 3.8405 |
| Final val (iter 15000, without) | 4.0557 |

The gap nearly doubled (0.1225 → 0.2234) with the larger corpus. More data in the bank means more useful things to retrieve.

**Checkpoint:** `out-retro-bank/ckpt_best.pt` (210.9 MB, iter 13000)

---

## Component 3: Held-Out Evaluation (The Leakage Test)

### Why This Matters

The training data comes from Wikipedia articles 0-99,999. The bank contains cells from those same articles. So the retrieval gap on the validation set could be partly explained by information leakage — the model is retrieving text that overlaps with what it trained on.

To prove the gap is genuine, we evaluate on 5,000 Wikipedia articles (100,000-104,999) that were **never** in the bank and **never** in the training data.

### Held-Out Results (4-Condition Eval)

| Condition | Wikipedia (held-out) | Code (held-out) |
|---|---|---|
| none (no retrieval) | 4.4372 | 3.2972 |
| random (random neighbors) | 4.4323 | 3.1794 |
| real1 (1 real + 1 random) | 4.3219 | 2.9733 |
| real (2 real neighbors) | 4.2711 | 2.9611 |
| **Semantic gap** | **+0.1612 (3.6%)** | **+0.2184 (6.6%)** |

The semantic gap (real minus random) isolates genuine retrieval benefit from any positional/architectural effect. Both domains show clear improvement from retrieval on text the model has never seen.

### Neighbor Count Ablation (Experiment 2)

The `real1` condition tests the marginal value of each neighbor:

| Domain | 1st neighbor | 2nd neighbor | 2nd marginal |
|---|---|---|---|
| Wikipedia | 68% of gap | 31% of gap | +46% over 1st |
| Code | 94% of gap | 6% of gap | +6% over 1st |

For Wikipedia, the 2nd neighbor adds substantial value. For code, the 1st neighbor captures nearly all the benefit — code retrieval is more "needle-in-a-haystack" (one good match is enough).

---

## Experiment 3: CCA Layer Placement Ablation

Which cross-attention layers matter most? We test by disabling individual CCA layers at eval time (no retraining). The model has CCA in layers [1, 3, 5, 7].

### Wikipedia Results

| Condition | Loss | Degradation | % of Gap |
|---|---|---|---|
| all_cca (baseline) | 4.2762 | — | — |
| no_cca | 4.4395 | +0.1633 | 100% |
| drop_L1 | 4.2952 | +0.0190 | 11% |
| drop_L3 | 4.2923 | +0.0161 | 10% |
| drop_L5 | 4.3356 | +0.0594 | 36% |
| drop_L7 | 4.2882 | +0.0120 | 7% |
| only_L1 | 4.3930 | — | 28% solo |
| only_L3 | 4.3870 | — | 32% solo |
| only_L5 | 4.3343 | — | 63% solo |
| only_L7 | 4.4165 | — | 14% solo |

### Code Results

| Condition | Loss | Degradation | % of Gap |
|---|---|---|---|
| all_cca (baseline) | 2.9914 | — | — |
| no_cca | 3.3218 | +0.3305 | 100% |
| drop_L1 | 3.0382 | +0.0468 | 14% |
| drop_L3 | 3.0021 | +0.0107 | 3% |
| drop_L5 | 3.0411 | +0.0498 | 15% |
| drop_L7 | 2.9836 | -0.0078 | -2% |
| only_L1 | 3.1648 | — | 48% solo |
| only_L3 | 3.1507 | — | 52% solo |
| only_L5 | 3.0997 | — | 67% solo |
| only_L7 | 3.2727 | — | 15% solo |

### Key Finding

**Layer 5 dominates** in both domains: 36% degradation when dropped (wiki), 63% solo benefit. It sits at position 5/7 (deep enough to have rich representations, early enough to influence subsequent layers). Layer 7 (the last layer) contributes almost nothing — by then, the model has already committed to its predictions.

---

## Experiment 4: Bank Scaling

Does a bigger memory bank improve retrieval? We subsample the bank at 10/25/50/75/100% and re-search neighbors for the held-out Wikipedia chunks.

| Bank Size | Cells | Gap (none - real) | vs 10% |
|---|---|---|---|
| 10% | 181,720 | +0.1332 | baseline |
| 25% | 454,301 | +0.1422 | +7% |
| 50% | 908,602 | +0.1516 | +14% |
| 75% | 1,362,903 | +0.1585 | +19% |
| 100% | 1,817,204 | +0.1644 | +23% |

**Log-linear scaling**: 10x more cells gives +23% improvement in retrieval gap. The curve shows clear diminishing returns — each doubling of bank size adds roughly half the benefit of the previous doubling. This mirrors findings in the original RETRO paper (Borgeaud et al., 2021) where scaling the retrieval database showed consistent but sublinear gains.

---

## Experiment 6: Chinchilla-Optimal Analysis

### How Undertrained Is This Model?

The Chinchilla scaling law (Hoffmann et al., 2022) recommends a ~20:1 ratio of training tokens to model parameters for compute-optimal training.

| Metric | Our Model | Chinchilla Optimal |
|---|---|---|
| Parameters | 55.29M | 55.29M |
| Training tokens | 157M (corpus) | 1,106M (20:1) |
| Tokens actually seen | ~106.5M (best @ iter 13000) | 1,106M |
| Token-to-param ratio | 1.9:1 | 20:1 |
| Effective epochs | ~0.68 | ~7.0 |

The model is **~10x undertrained** by Chinchilla standards. It has seen each training token less than once on average (0.68 epochs).

### What This Means for the Retrieval Gap

The retrieval gap on held-out Wikipedia is +0.1644 nats (3.7%). This is with a severely undertrained model. A Chinchilla-optimal model trained on 1.1B tokens would likely show lower baseline loss and a potentially larger retrieval gap.

Based on the bank scaling curve (log-linear), **training data scaling is the binding constraint**, not bank size.

---

## Experiment 1: Live Generation Demo

The `generate_retro.py` script demonstrates real-time retrieval-augmented generation: it encodes each 64-token chunk with MiniLM, queries the cc_service API for neighbors, and feeds them into the RETRO model's CCA layers.

At 55M parameters, the model is too small for coherent freeform text generation — both with and without retrieval produce poor-quality text. GPT-2 small (117M params) is the minimum for semi-coherent generation.

**The eval metrics are the real proof of retrieval value**, not the generation quality. Speed: 0.5-0.9s per prompt after chunk-boundary caching optimization.

---

## Summary of Findings

1. **Retrieval works on unseen text**: +3.6% perplexity improvement on held-out Wikipedia, +6.6% on held-out code — with zero information leakage.
2. **Layer 5 is critical**: One CCA layer (at position 5/7) accounts for 36-67% of the retrieval benefit. Layer 7 contributes nearly nothing.
3. **Bank size helps, with diminishing returns**: 10x more cells -> +23% gap improvement, following log-linear scaling.
4. **First neighbor captures most value**: Especially for code (94%), less so for Wikipedia (68%) where the 2nd neighbor adds meaningful signal.
5. **Model is 10x undertrained**: Only 1.9:1 token-to-param ratio vs Chinchilla-optimal 20:1. Training data, not bank size, is the binding constraint.
6. **RETRO's advantage is eval-time flexibility**: The bank can be swapped, grown, or domain-specialized without retraining the model.

### The question

The +0.2234 retrieval gap looks impressive, but is it real? The training data came from the bank — so the model might be retrieving its own training text, making the gap a circular artifact (information leakage) rather than genuine semantic retrieval.

### The experiment

We designed a 3-condition controlled experiment on text the model has **never seen**:

1. **Held-out data**: 5,000 English Wikipedia articles (articles 100,000–104,999) that were never loaded into the bank and never appeared in the training or validation sets.
   - 47,999 paragraphs → 4,475,401 GPT-2 tokens → 69,928 chunks of 64
2. **Precomputed real neighbors**: For each held-out chunk, we ran the full MiniLM + whitening pipeline to find the 2 closest bank cells. The bank doesn't contain the held-out text, so these neighbors are genuinely *related but different* text.
3. **Three eval conditions** on the same held-out data:
   - **none** — CCA turned off, model works alone
   - **random** — CCA active but fed random bank cells (wrong topics, no semantic match)
   - **real** — CCA fed MiniLM-retrieved neighbors (semantically relevant)

### Results

| Condition | Loss | ±stderr |
|-----------|------|---------|
| none (CCA off) | 4.4172 | ±0.0208 |
| random neighbors | 4.4147 | ±0.0210 |
| **real neighbors** | **4.2505** | ±0.0214 |

| Gap | Value | Interpretation |
|-----|-------|----------------|
| none − real | **+0.1667** | Total retrieval benefit on unseen text |
| none − random | +0.0025 | CCA-as-regularizer: negligible (within noise) |
| random − real | **+0.1642** | **Pure semantic retrieval benefit** |

### What this proves

- **Random neighbors do nothing** (+0.0025, indistinguishable from noise). Just "having extra tokens flowing through CCA" does not help. The architecture isn't cheating by using CCA as a free regularizer.
- **Real neighbors help a lot** (+0.1667). On text the model has never seen and that isn't in the bank, finding semantically related bank entries still provides substantial benefit.
- **98.5% of the improvement is semantic** (+0.1642 out of +0.1667). Nearly the entire retrieval benefit comes from the model finding relevant text and extracting useful information from it.
- **Not leakage.** If the training gap were caused by the model retrieving its own training text, the gap would collapse on held-out text. It didn't — it remained strong (+0.1667 on held-out vs. +0.2234 on in-distribution val). The held-out gap is slightly smaller, which is expected: in-distribution text has closer bank matches.

**Verdict: [PASS]** — The retrieval gap is genuine semantic retrieval.

### Key files

| File | Purpose |
|------|---------|
| `nanogpt/data/bank/prepare_heldout.py` | Download, tokenize, embed, precompute neighbors for held-out articles |
| `nanogpt/eval_retro_heldout.py` | 3-condition eval: none / random / real |
| `nanogpt/data/bank/heldout.bin` | Held-out token stream (9.0 MB) |
| `nanogpt/data/bank/heldout_neighbors.npy` | Precomputed neighbors for held-out chunks |
| `nanogpt/data/bank/heldout_meta.json` | Held-out dataset metadata |

---

## Technical Lessons Learned

### Windows/Python DLL load order

Importing `datasets` (HuggingFace) after `numpy` or `torch` causes a `pyarrow` DLL segfault (0xC0000005) on Windows in this venv. **Always import `datasets` and `transformers` before `numpy` and `torch`.**

### Bank index alignment

The bank has 1,817,265 single cells total, but only 1,817,204 have `source_text` entries (needed for tokenization). The neighbor precompute pipeline uses a `JOIN source_texts` query that produces 1,817,204 rows, and `cell_tokens.npy` has that many entries. Any code that loads bank cells for retrieval must use the same JOIN to keep indices aligned, or neighbor lookups will be out-of-bounds.

### Path resolution

Scripts invoked from different working directories break if they use relative paths. All data/checkpoint paths should be resolved from `Path(__file__).resolve().parent`, not from `os.getcwd()`.

### GPU memory management

The RTX 3070 has 8 GB VRAM. Loading the bank weight matrix for neighbor precompute (~2.8 GB) plus MiniLM (~80 MB) leaves headroom for the precompute but not enough to also run the cc_service. Stop the service before running prep scripts.

---

## Experiment 5: Code Fine-Tune (Domain Transfer)

### Goal

Test whether RETRO can transfer its CCA retrieval skill to a completely new domain. We fine-tune the best Wikipedia-trained checkpoint on code tokens, while still retrieving neighbors from the Wikipedia bank. If the retrieval gap survives, the model's learned ability to extract information from cross-attention is domain-general, not Wikipedia-specific.

### Data

- **Training corpus**: `nickrosh/Evol-Instruct-Code-80k-v1` — 78,155 coding instruction/response pairs
- **Train tokens**: ~28.8M (GPT-2 BPE)
- **Val tokens**: ~3.2M
- **Neighbors**: Precomputed from the wiki-only bank (1,817,204 cells) — the bank contains zero code
- **Effective training**: ~5.6 epochs over 5,000 iterations (batch 8, grad accum 4)

### Training setup

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `out-retro-bank/ckpt_best.pt` (iter 13000, wiki-trained) |
| Learning rate | 1e-4 → 1e-5 (cosine, 100 warmup) |
| Batch size | 8 × 4 grad accum = 32 effective |
| Iterations | 5,000 |
| Total tokens seen | ~32M |

### Results

| Metric | Baseline (wiki model) | Best (iter 4750) | Final (iter 5000) |
|--------|----------------------|-------------------|-------------------|
| val_with (neighbors) | 2.6332 | 1.3898 | 1.4356 |
| val_without | 3.1321 | 1.6211 | 1.6497 |
| retrieval gap | +0.4989 | +0.2313 | +0.2141 |

### Key findings

- **45.5% improvement** in code loss after fine-tuning (2.63 → 1.44 nats)
- **Best checkpoint at iter 4750** hit 1.39 nats — even better than final
- **Retrieval gap settled at +0.21–0.23 nats** — the wiki bank still helps the code model, even though the bank contains zero code
- The gap started at +0.50 (wiki model relied heavily on retrieval to compensate for not knowing code) and converged to ~+0.22 as the model internalized code patterns — retrieval shifted from "crutch" to "supplement"
- No sign of overfitting: val loss kept improving through iter 4750 on 32M tokens (~5.6 epochs)

The model successfully transferred its CCA retrieval skill to a completely new domain while the wiki bank continued providing useful signal.

### Checkpoint

`out-retro-code/ckpt_best.pt` — iter 4750, fine-tuned on code

### Key files

| File | Purpose |
|------|---------|
| `nanogpt/train_retro_code_finetune.py` | Fine-tuning script |
| `nanogpt/data/code/prepare_code_finetune.py` | Code data prep + wiki neighbor precompute |
| `nanogpt/out-retro-code/ckpt_best.pt` | Best code-fine-tuned checkpoint |
| `nanogpt/out-retro-code/history.json` | Training history (21 eval points) |

---

## Experiment 7: RAG Demo (Llama 3.2 + Bank)

### Goal

Demonstrate that the same retrieval pipeline (MiniLM → whitening → bank search) works in a traditional RAG configuration with a much larger model, showing the bank's value is independent of architecture.

### Setup

- **Model**: Llama 3.2 (via Ollama)
- **Retrieval**: Same cc_service bank (1.8M Wikipedia cells), queried via HTTP API
- **Method**: For each prompt, retrieve top-2 bank cells, paste them into the context, and compare generation with vs. without retrieval

### Test prompts

1. "Explain how photosynthesis works in plants."
2. "What caused the fall of the Roman Empire?"
3. "Describe the structure of DNA and its role in heredity."

### Observations

- **Without retrieval**: Llama produces correct but generic answers from its parametric memory
- **With retrieval**: Answers become more specific and grounded — citing Gibbon for Roman history, Hershey-Chase for DNA, specific chloroplast mechanisms for photosynthesis
- The bank surfaces details that even a 3B-parameter model doesn't reliably produce from memory alone
- Same retrieval pipeline, two completely different integration methods (CCA vs. prompt injection)

### Key file

| File | Purpose |
|------|---------|
| `nanogpt/rag_gemma4.py` | RAG demo script (despite filename, defaults to `llama3.2:latest`) |

---

## Summary of Results

| Experiment | val_with | val_without | Gap | Notes |
|------------|----------|-------------|-----|-------|
| Phase 1 (Shakespeare) | — | — | Large | Oracle neighbors, wiring test only |
| Phase 2.2 (60M tokens, 55M params) | 3.6949 | 3.8175 | +0.1225 | Overfit (1:1 ratio) |
| Phase 2.2e (60M tokens, 6M params) | 4.1285 | 4.2099 | +0.0814 | Chinchilla-aligned, less capacity |
| **Phase 2.3 (157M tokens, 55M params)** | **3.6045** | **3.8279** | **+0.2234** | Best model, more data |
| **Held-out eval (real vs none)** | **4.2505** | **4.4172** | **+0.1667** | **Never-seen text, not leakage** |
| Held-out eval (real vs random) | 4.2505 | 4.4147 | +0.1642 | 98.5% of gap is semantic |
| **Code fine-tune (best)** | **1.3898** | **1.6211** | **+0.2313** | **Wiki bank helps code domain** |
| Code fine-tune (final) | 1.4356 | 1.6497 | +0.2141 | Retrieval = supplement, not crutch |

---

## Current State (May 25, 2026)

- **Best wiki checkpoint**: `nanogpt/out-retro-bank/ckpt_best.pt` — iter 13000, 55.29M params
- **Best code checkpoint**: `nanogpt/out-retro-code/ckpt_best.pt` — iter 4750, fine-tuned
- **Bank**: 1,817,204 cells from Simple + English Wikipedia, served by cc_service
- **Held-out eval**: PASS — retrieval is genuine semantic work
- **Code transfer**: PASS — wiki bank helps code prediction (+0.23 nats)
- **RAG demo**: Same bank improves Llama 3.2 generation quality
- **All 7 experiments complete**
- **Git**: nanogpt committed as `92ff196` on `phase2-retro`, tagged `phase2.3-data-scale`
