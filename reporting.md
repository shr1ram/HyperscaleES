# Improving Learned KV Cache Eviction via Joint Model-Manager Co-Training

## Overview

The NAMM paper (Cetin et al., 2024) introduces a learned KV cache eviction policy that compresses transformer memory by 75% while maintaining or improving performance. However, the base transformer model is frozen throughout NAMM training — the eviction policy must adapt to whatever attention patterns the pretrained model happens to produce, with no guarantee that those patterns are optimal for operating under a compressed cache.

**This project's hypothesis:** if we jointly train the model's weights (via LoRA) alongside the NAMM eviction policy, the model can learn to produce attention patterns that are more compatible with cache compression. The model and the cache manager co-adapt, leading to better eviction performance than training the cache manager alone.

We use **EGGROLL** (Sarkar et al., 2025) as the optimizer for the model's LoRA weights — it is gradient-free, which means we never need to backpropagate through the eviction decisions. We use **CMA-ES** (as in the original NAMM paper) for the eviction policy's parameters. Both optimizers share a single fitness signal: task performance under cache compression.

The target benchmark is **QASPER** (Dasigi et al., 2021), a question-answering dataset over scientific papers with ~3,000-5,500 token contexts — long enough to require meaningful cache management but short enough to be feasible on a single GPU.

---

## 1. NAMM: The Research Target

### What NAMM does

[NAMM](https://arxiv.org/abs/2410.13166) (Neural Attention Memory Model) is a tiny neural network (~4,000 parameters) that acts as a KV cache eviction policy. It observes how the transformer's attention heads use each cached token, and decides whether to keep or evict that token. The physical mechanism is simple: tokens with negative scores are removed from the KV cache entirely.

### Architecture: BAM (Backward Attention Memory)

The BAM network is a single self-attention head with **counter-causal (backward) masking** — each token can attend to tokens that are *newer* than itself, but not older ones. This creates competition: older tokens must justify their retention against newer arrivals.

- Hidden dimension: 16
- Input: 25-dimensional feature per cached token (17 spectral features + 8 positional)
- Output: scalar score per token. If score < 0, the token is evicted.
- Total parameters: ~4,000, shared across all transformer layers.

### Feature extraction pipeline

Every 512 new tokens, NAMM runs this pipeline on each cached token:

1. **Extract attention columns** — for each cached token, gather how much the recent 512 queries attended to it.
2. **Short-Time Fourier Transform** — convert the attention signal into a spectrogram (17 frequency bins per time window, using a Hann window of size 32, stride 16).
3. **Exponential Moving Average** — compress the spectrogram across time into a single 17-dim vector (gamma = 0.99^16 ≈ 0.85), with history carried across cycles.
4. **Positional encoding** — append an 8-dim sinusoidal encoding of token age.
5. **BAM classifier** — produces a keep/evict score per token.

### How NAMM is trained (baseline)

NAMM is trained with **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy). CMA-ES maintains and adapts a full covariance matrix over the parameter space, which is feasible at ~4,000 parameters. The base transformer is completely frozen. Training uses a 3-stage curriculum across tasks (PassageRetrieval, DuReader, NarrativeQA) with population size 32.

### Key results on frozen models

- **LongBench**: 75% cache reduction with 1.11x performance (29.33 vs 28.86 baseline).
- **InfiniteBench (200K tokens)**: 60% cache reduction with 10.45x performance improvement.
- **Zero-shot transfer**: transfers across model scales (8B→70B), modalities (text→vision), and domains (NLP→RL).
- **Sub-linear cache scaling**: longer contexts get compressed more aggressively.

### The limitation this project addresses

NAMM treats the base model as a fixed black box. The eviction policy must work with whatever attention patterns the pretrained model produces. But there is no reason those patterns are optimal for operating under a compressed cache. Consider:

- A model might distribute important information across many tokens, making it hard to evict any without quality loss. If the model could learn to concentrate information into fewer tokens, NAMM could compress more aggressively.
- A model might rely on subtle long-range attention patterns that break when intermediate tokens are evicted. If the model could learn more robust attention patterns — ones that degrade gracefully under eviction — NAMM could be more aggressive.
- A model might produce attention distributions where the "importance" of a token is ambiguous from the attention signal alone. If the model could learn to make its attention patterns more informative for the eviction policy, NAMM could make better decisions.

**Joint training allows the model and the cache manager to find a cooperative equilibrium that neither can reach alone.**

---

## 2. EGGROLL: The Optimizer for Joint Training

### Why EGGROLL

We need a gradient-free optimizer for the model's weights because:
1. The NAMM eviction decision is binary (keep/evict) — non-differentiable. We cannot backpropagate through it.
2. EGGROLL is already implemented in HyperscaleES and proven to work for LoRA fine-tuning of language models.
3. Both EGGROLL (for model weights) and CMA-ES (for NAMM) are evolutionary methods that use fitness signals. They can share the same fitness evaluation: task performance under cache compression.

### How EGGROLL works

[EGGROLL](https://arxiv.org/abs/2511.16652) (Evolution Guided General Optimisation via Low-rank Learning) is a gradient-free training method that replaces backpropagation with evolutionary strategies using low-rank noise perturbations.

**Core mechanism:** Standard ES estimates a gradient by evaluating a population of perturbed parameter vectors:

```
grad ≈ (1/N) * sum_i [ fitness_i * epsilon_i ]
```

EGGROLL's key insight: **factor each perturbation as a low-rank outer product**. Instead of sampling a full `(a x b)` noise matrix per weight, sample two small vectors `A_i ∈ R^(a x r)` and `B_i ∈ R^(b x r)` where `r` is the LoRA rank (typically 1). The perturbation is `(sigma / sqrt(r)) * A_i @ B_i^T`. This reduces per-perturbation memory from O(ab) to O(r(a+b)).

**On-the-fly generation:** Perturbations are never stored. Each population member's noise is deterministically regenerated from a counter-based RNG key seeded by `(base_key, epoch, thread_id)`. The base model parameters are shared (one copy), and each thread's noise is generated on the fly during the matmul.

**Antithetic sampling:** Population members are paired — even threads use `+sigma`, odd threads use `-sigma` with the same noise vectors, halving variance.

**Update:** After rollouts, fitness scores are z-normalized, the same LoRA perturbations are regenerated from seeds, and a fitness-weighted aggregate gradient is applied via an optimizer (SGD by default). Despite rank-1 perturbations, the aggregated update across N members is effectively full-rank when N > min(a, b).

### Key properties relevant to joint training

- **No backpropagation through the model** — fitness is the only signal. This means NAMM's non-differentiable eviction decisions pose no problem.
- **On-the-fly noise** — perturbations are generated from seeds, not stored. Memory cost is dominated by per-thread model state, not parameters.
- **Shared fitness signal** — the same task-level score (e.g., QASPER F1) that evaluates the model's answer quality also reflects how well the model cooperates with the eviction policy.

---

## 3. The Bottleneck: Per-Thread KV Cache in Evolutionary Training

### Why transformers are harder than RWKV for ES

EGGROLL was originally designed around RWKV, a recurrent architecture with **fixed-size state** regardless of sequence length:

```
RWKV state per thread: (n_layer, 1+head_size, n_embd)
Example (RWKV 0.1B):   (12, 65, 768) = ~1.3 MB  (constant for any seq_len)
```

Standard transformers (including TinyLlama) use **KV caches** that grow with sequence length:

```
KV cache per thread: (n_layer, 2, max_seq_len, n_kv_heads, head_dim)
TinyLlama @ 256:     (22, 2, 256, 4, 64)   = ~5.8 MB per thread
TinyLlama @ 2048:    (22, 2, 2048, 4, 64)   = ~46 MB per thread
TinyLlama @ 4096:    (22, 2, 4096, 4, 64)   = ~92 MB per thread
```

In the EGGROLL generation loop, `jax.vmap` maps across the population. Each thread has different LoRA noise, generates different tokens, and therefore has a unique KV cache. The cache is inside the vmapped scan carry and cannot be shared.

### What IS NOT the bottleneck

- **Model parameters**: shared across all threads (one copy). TinyLlama 1.1B = 2.2 GB in bf16.
- **Noise perturbations**: generated on the fly from RNG keys. Zero storage.
- **Fitness computation**: runs on CPU, cheap.
- **LoRA update aggregation**: ~789K LoRA params. Trivial.

### Memory scaling

| Sequence length | Per-thread state | x256 threads | x1024 threads |
|-----------------|------------------|--------------|---------------|
| 256 (toy tasks) | 5.8 MB | 1.5 GB | 5.9 GB |
| 2048 (GSM8K) | 46 MB | 11.8 GB | 47.1 GB |
| 4096 (QASPER) | 92 MB | 23.5 GB | 94.2 GB |

On a 24 GB GPU, at sequence length 4096 even 256 threads would consume the entire budget.

### Compute scaling

Attention cost per token at position `t` is O(t). Total across a generation of length L is O(L^2):

| Sequence length | Relative attention cost (vs 256) |
|-----------------|----------------------------------|
| 256 | 1x |
| 512 | 4x |
| 1024 | 16x |
| 2048 | 64x |
| 4096 | 256x |

### How NAMM alleviates this (as a side benefit)

While the primary goal is improving NAMM's eviction quality, cache compression also directly reduces the ES bottleneck. With 75% eviction, a 4096-token context uses only ~1024 cache entries, bringing per-thread state and attention cost back to manageable levels:

| Property | Without NAMM (4096) | With NAMM (4096 → ~1024 effective) |
|----------|--------------------|------------------------------------|
| Per-thread state | 92 MB | ~23 MB |
| x256 threads | 23.5 GB | ~5.9 GB |
| Attention cost (relative) | 256x | ~16x |

This is a secondary benefit — the research contribution is the co-training, not the memory savings.

---

## 4. Difficulties in Joint EGGROLL + NAMM Training

### 4.1 Credit assignment

When fitness improves during joint training, was it because:
- (a) The LoRA update made the model better at the task?
- (b) The LoRA update made the model's attention patterns more compatible with NAMM's eviction?
- (c) The NAMM update found a better eviction policy?
- (d) Some combination of all three?

EGGROLL and CMA-ES each follow their own fitness gradient and cannot distinguish between these cases. This is not necessarily a problem — the system as a whole improves — but it means we cannot independently measure the model's contribution vs the cache manager's contribution without ablation experiments.

### 4.2 Co-evolution instability (moving target problem)

NAMM adapts its eviction policy to the model's current attention patterns. EGGROLL simultaneously changes those attention patterns via LoRA updates. If they move at different rates:

- NAMM learns to handle attention pattern A.
- EGGROLL shifts the model to pattern B.
- NAMM re-adapts to B.
- EGGROLL shifts to C.

This oscillation can prevent convergence to a stable equilibrium. Mitigations:
- **Slow LoRA updates**: use small sigma (1e-3) and conservative learning rate so attention patterns shift gradually, giving NAMM time to track.
- **Periodic NAMM re-training**: run several NAMM CMA-ES generations after every N EGGROLL epochs.
- **Alternating freeze**: alternate between freezing NAMM (EGGROLL-only epochs) and freezing LoRA (NAMM-only epochs). This guarantees each optimizer faces a stationary target during its update.

### 4.3 Different optimizer scales

| Property | EGGROLL (LoRA) | CMA-ES (NAMM BAM) |
|----------|---------------|-------------------|
| Parameters | ~789,000 | ~4,000 |
| Population size | 256-1024 | 32 |
| Optimizer | EGGROLL (rank-1 ES + SGD) | CMA-ES (full covariance) |
| Per-generation cost | Full model forward pass | Feature extraction + tiny classifier |
| Sigma schedule | Fixed or slow decay | Adaptive (CMA-ES self-tunes) |

CMA-ES maintains a full covariance matrix of size O(d^2). At 4,000 params this is 16M entries — feasible. At 789,000 params this would be 622 billion entries — impossible. So we cannot use a single optimizer for both. The two optimizers must run with different population sizes, update frequencies, and sigma schedules while sharing a fitness signal.

### 4.4 Per-thread attention divergence

NAMM's input features come from attention matrices. During EGGROLL training, each population member has different LoRA noise, producing different attention patterns. This creates a choice:

- **Per-thread NAMM** (faithful): extract attention features and run BAM for each population member independently. Expensive — it means running the STFT + BAM pipeline 256+ times per eviction cycle. But eviction decisions are tailored to each member's actual attention.
- **Shared NAMM** (approximate): run NAMM on the base (unperturbed) model's attention patterns and apply the same eviction mask to all threads. Cheap, but eviction decisions may be suboptimal for perturbed models. However, since sigma is small (1e-3), attention patterns across the population are similar.
- **Sampled NAMM** (compromise): run per-thread NAMM on a small subsample of the population (e.g., 8 threads), average the eviction masks, and apply to all. Balances cost and accuracy.

The shared approach is the practical starting point. As LoRA updates accumulate and the model drifts from its initialization, per-thread NAMM may become necessary.

### 4.5 Fitness signal is entangled

In standalone NAMM training, fitness measures "model performance with this eviction policy vs without." In standalone EGGROLL, fitness measures "model performance with this LoRA perturbation." In joint training, the fitness is a single number reflecting both contributions simultaneously.

A bad LoRA perturbation paired with a good eviction policy produces the same low fitness as a good perturbation with bad eviction. Neither optimizer can distinguish its own contribution. This is inherent to the joint optimization and is unlikely to be solved without additional signal decomposition (e.g., running some evaluations without NAMM as a control).

### 4.6 Static shape constraints in JAX

The `jax.lax.scan`-based generation loop requires static shapes for the carry (state). NAMM's eviction changes the effective cache size dynamically. Options:

- **Logical masking**: keep the physical cache at max_seq_len but maintain a validity mask. Evicted entries are masked out in attention but the array shape doesn't change. This preserves static shapes and provides compute savings (softmax over fewer valid entries) but no memory savings.
- **Compaction**: physically remove evicted entries and compact the cache. Saves memory but requires dynamic shapes, incompatible with `jax.lax.scan`.

Logical masking is the pragmatic choice for the initial implementation.

### 4.7 Eviction periodicity in the generation loop

NAMM runs every 512 new tokens. The EGGROLL generation loop processes tokens one at a time via `jax.lax.scan`. Injecting periodic NAMM calls requires either:
- A conditional inside the scan body (`jax.lax.cond` every 512 steps).
- Chunking generation into 512-token segments with NAMM eviction between chunks.

The chunked approach is cleaner: run 512 steps of `jax.lax.scan`, apply NAMM eviction to the cache, then run the next 512 steps.

---

## 5. QASPER as the Target Benchmark

### Why QASPER

The NAMM paper uses **LongBench** (36 tasks, contexts up to 32K tokens) as its primary benchmark. LongBench is comprehensive but its long contexts make it impractical for initial joint training experiments on a single GPU.

[QASPER](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al., 2021) is a question-answering dataset over NLP research papers:

- **1,585 papers**, **5,049 questions**, split into train (888) / validation (281) / test (416).
- **Moderate context**: papers average ~4,100 words ≈ 3,000-5,500 tokens. Long enough to require meaningful cache management but short enough to be feasible on a single GPU.
- **Smooth fitness signal**: token-level F1 between predicted and gold answers. Continuous in [0, 1], much better for ES than binary accuracy.
- **Mixed answer types**: extractive spans, free-form text, yes/no, and "unanswerable" — tests genuine comprehension.
- **Standard metric**: F1 is the established QASPER evaluation metric, enabling comparison with published baselines.

### QASPER vs LongBench

| Property | LongBench | QASPER |
|----------|-----------|--------|
| Context length | 8K-32K tokens | 3K-5.5K tokens |
| KV cache @ 256 threads | 47-188 GB | 6-12 GB |
| Feasible on 24 GB GPU | No | Yes (with truncation or NAMM) |
| Fitness signal | Varies by task | Token F1 (smooth, continuous) |
| Number of tasks | 36 | 1 (focused) |

### Implementation

A QASPER task class following the existing `BanditTask` pattern in `llm_bandits.py`:

**Prompt construction**: paper title + abstract + full text sections (truncated to fit), followed by the question:

```
User: Read the following paper and answer the question.

Title: {title}
Abstract: {abstract}
{full_text sections, truncated}

Question: {question}
Assistant: <think
```

**Fitness scoring**: token-level F1 between the generated answer (after `</think>`) and gold. For multiple annotator answers, take max F1 across golds.

```python
def token_f1(prediction, gold):
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

**Flattening**: one example per (paper, question) pair, with gold answer being the best available annotation.

---

## 6. Experimental Plan

### Experiment 1: Baselines (no joint training)

Establish independent baselines to measure the improvement from co-training:

| Condition | What we measure |
|-----------|----------------|
| **TinyLlama, no NAMM, no EGGROLL** | Base model QASPER F1 (zero-shot) |
| **TinyLlama + EGGROLL, no NAMM** | F1 after LoRA fine-tuning (full KV cache, truncated context) |
| **TinyLlama + NAMM, no EGGROLL** | F1 with learned eviction (frozen model, CMA-ES for BAM only) |

These three numbers define the landscape. EGGROLL-only tells us how much LoRA helps on the task. NAMM-only tells us how well eviction works on a frozen model. The joint system should improve on NAMM-only (the primary claim) and ideally on EGGROLL-only as well.

### Experiment 2: Joint training

Run EGGROLL and CMA-ES together with shared fitness:

- **Alternating schedule**: N EGGROLL epochs (frozen NAMM), then M CMA-ES generations (frozen LoRA), repeat.
- **Shared NAMM**: use base model attention patterns for eviction (approximate, cheap).
- **Metric**: QASPER F1 under cache compression, compared to NAMM-only baseline.

Key questions to answer:
- Does F1 improve over NAMM-only at the same compression rate?
- Can the joint system achieve higher compression (e.g., 85% eviction vs 75%) at the same F1?
- Does the model's attention pattern measurably change to accommodate eviction?

### Experiment 3: Ablations

| Ablation | Question |
|----------|----------|
| Joint training vs alternating | Does simultaneous optimization outperform alternating? |
| Shared vs per-thread NAMM | Does per-thread eviction improve fitness at the cost of compute? |
| LoRA rank 1 vs 2 vs 4 | Does higher rank help the model adapt its attention patterns? |
| Eviction rate sweep | At what compression rate does joint training diverge from NAMM-only? |
| Sigma schedule | Does slower sigma decay help co-training stability? |

### Experiment 4: Scaling (if results are positive)

- Move to LongBench tasks with 8K-32K contexts.
- Test on larger models (RWKV-7 1.5B with KV cache adapter, or a larger LLaMA variant).
- Evaluate zero-shot transfer of jointly-trained NAMM to unseen models.

---

## 7. Implementation Roadmap

### Phase 1: TinyLlama baseline (done)
- TinyLlama 1.1B integrated into HyperscaleES with bucketed KV cache state.
- Verified on fastzero with EGGROLL training.
- Tests: parameter shapes, forward pass, KV cache tracking, HuggingFace numerical match, EGGROLL noiser init + update.

### Phase 2: QASPER task
- Add `QASPERTrain` / `QASPERTest` task classes to `llm_bandits.py`.
- Flatten dataset, construct prompts with truncated paper context.
- Implement token-F1 fitness scoring.
- Establish Experiment 1 baselines: zero-shot F1 and EGGROLL-only F1 (truncated context, bucket=2048, batch=128).

### Phase 3: NAMM on frozen TinyLlama
- Port NAMM's BAM network (~4K params) to JAX.
- Implement STFT feature extraction from attention matrices.
- Add logical masking to KV cache for eviction (static shapes, compute savings).
- Chunk the generation loop for periodic NAMM execution (every 512 tokens).
- Train NAMM with CMA-ES on QASPER with frozen TinyLlama.
- Establish NAMM-only baseline (Experiment 1, third row).

### Phase 4: Joint EGGROLL + NAMM co-training
- Implement alternating optimization schedule.
- Shared NAMM eviction across EGGROLL population (base model attention patterns).
- Run Experiment 2: joint training on QASPER.
- Compare against all three baselines from Experiment 1.

### Phase 5: Analysis and ablations
- Run Experiment 3 ablations.
- Analyze attention pattern changes: do attention distributions become sparser or more eviction-compatible after joint training?
- Measure whether joint-trained NAMM transfers to unseen prompts / contexts better than NAMM-only.

### Phase 6: Scaling
- Run Experiment 4 on LongBench / larger models if QASPER results support the hypothesis.

---

## 8. Critical Assessment

### Novelty

The core idea — co-evolving model weights (via EGGROLL LoRA) alongside a NAMM eviction policy — has not been done. NAMM always freezes the model. EGGROLL never touches cache management. The literature search confirms the gap is real.

However, the novelty gap is narrower than it first appears. Closely related work occupies adjacent territory:

| Paper | Relationship | Key difference |
|-------|-------------|----------------|
| **MatryoshkaKV** (ICLR 2025) | Jointly trains LoRA + KV cache *dimension* compression with gradients | Compresses feature dimensions, not token-level eviction; gradient-based |
| **Attention-Gate** (2024) | Learns a differentiable eviction gate alongside model weights | Gradient-based; requires differentiable relaxation of the eviction decision |
| **EvolKV** (EMNLP 2025) | Uses CMA-ES to optimize per-layer cache budgets | Evolutionary + cache, but does not adapt model weights |
| **Learning to Evict / KVP** (Feb 2026) | RL-trained per-head eviction agents | RL, not ES; inference-time only, no weight adaptation |
| **ESSA** (Jul 2025) | ES for LoRA alignment at scale (EGGROLL follow-up) | Validates ES for LoRA but has no cache component |

The unique claim is: gradient-free co-optimization of token-level eviction + model weights, using the non-differentiability of the binary eviction decision as a feature rather than an obstacle. This is publishable.

### Feasibility concerns

**What works well:**
- EGGROLL on TinyLlama compiles and runs (test suite proves this).
- Per-thread memory at bucket=256 is manageable (~5.5 MB/thread).
- QASPER F1 as a fitness signal is smooth and implementable within the existing task framework.

**What is concerning:**

1. **TinyLlama 1.1B is probably too weak for QASPER.** A 1.1B model reading a scientific paper and answering questions about it will likely produce near-zero F1 in zero-shot. EGGROLL LoRA with rank-1 perturbations may not have enough capacity to bridge that gap. The fitness signal could be too sparse for ES — most population members score 0, z-normalization of all zeros is uninformative.

2. **The sequence length problem is real.** QASPER papers are 3K-5.5K tokens. The current bucket is 256. Even at 4096, on a 24 GB GPU with 256 threads, the KV cache alone consumes 23.5 GB. Dropping to ~64 threads kills ES population diversity.

3. **Phases 1-3 must all work individually before Phase 4 (joint training) is meaningful.** Phase 1 is done. Phase 3 (porting NAMM's BAM network + STFT pipeline to JAX) is a substantial engineering effort — the feature extraction pipeline is non-trivial.

4. **Rank-1 LoRA perturbations may be too weak to shift attention patterns.** At sigma=1e-3, the model's attention distributions may not change enough for NAMM to "notice" a difference. The hypothesis — that the model learns more eviction-compatible attention — requires measurably different attention patterns, which small perturbations may not produce.

### Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Near-zero QASPER F1 on TinyLlama | No fitness signal for ES | Prove concept on fastzero/gsm8k first; scale to larger model later |
| KV cache OOM at long sequences | Cannot run QASPER at full context | NAMM eviction itself reduces cache; start with truncated papers |
| NAMM port too complex | Delays the core experiment | Use a simpler learned eviction policy (small MLP, ~100 params) as a stand-in |
| Co-training shows no improvement | Negative result | Still publishable as a negative result with proper analysis; ablations clarify why |
| Alternating optimization oscillates | No convergence | Slow LoRA updates (small sigma), freeze/thaw schedule, inner-loop NAMM convergence |

### Recommended course correction

**Compress the plan. Get to the co-training experiment as fast as possible.**

The current roadmap front-loads too much engineering (QASPER task, full NAMM port) before testing whether the core hypothesis holds. Instead:

1. **Prove the concept on a toy task first.** Use `fastzero` or a short-context task where TinyLlama produces nonzero fitness. Implement a simple eviction policy (attention-sum MLP, ~100 params, CMA-ES trained). Show that EGGROLL + eviction > eviction alone. This can be done in days, not weeks.

2. **If the concept works on toy tasks, invest in NAMM and QASPER.** The full STFT + BAM pipeline and QASPER integration are justified once there is evidence that co-training helps.

3. **If TinyLlama is too weak for QASPER, be explicit about it.** Frame TinyLlama as a proof-of-concept for the co-training mechanism. The interesting results will come from models that can actually do the task (3B+).

4. **Consider the gradient-based alternative as a comparison.** Several recent papers (Attention-Gate, MatryoshkaKV) show gradient-based co-training works. EGGROLL's value is when you cannot use gradients (binary eviction). A relaxed (Gumbel-softmax) eviction baseline with standard LoRA fine-tuning would be a strong comparison — if EGGROLL matches or beats it, that validates the gradient-free approach.

### Verdict

The direction is sound and the novelty is real. The risk is spending months building infrastructure and discovering that TinyLlama + QASPER does not produce a strong enough signal to validate the hypothesis. The fastest path to a publishable result is: prove co-training helps on a simple task, then scale up.

---

## 9. TPU Resources and Revised Plan

### Available compute

Access to Google Cloud TPU quota (30 days, project `statistical-nlp`):

| Resource | Chips | HBM/chip | Total HBM | bf16 TFLOPS/chip | Type |
|----------|-------|----------|-----------|-------------------|------|
| 64x v6e (europe-west4-a) | 64 | 32 GB | 2,048 GB | 918 | spot |
| 64x v6e (us-east1-d) | 64 | 32 GB | 2,048 GB | 918 | spot |
| 64x v5e (us-central1-a) | 64 | 16 GB | 1,024 GB | 197 | spot |
| 64x v5e (europe-west4-b) | 64 | 16 GB | 1,024 GB | 197 | spot |
| 32x v4 (us-central2-b) | 32 | 32 GB | 1,024 GB | 275 | spot |
| 32x v4 (us-central2-b) | 32 | 32 GB | 1,024 GB | 275 | on-demand |

For context, the 3090 Ti has 24 GB of VRAM. A single v6e-64 pod has 2 TB of HBM — an 85x increase.

### What this unlocks

The three biggest feasibility concerns from Section 8 are resolved:

**1. Model size: TinyLlama 1.1B was too weak for QASPER. Now we can run 8B.**

With 2 TB HBM on a v6e-64 pod:

| Model | Params (bf16) | KV cache @ 4096 seq, 1024 threads | Total | Fits on v6e-64? |
|-------|--------------|--------------------------------------|-------|-----------------|
| TinyLlama 1.1B | 2.2 GB | 92 GB | ~95 GB | Easily |
| Llama 3.2 3B | 6 GB | ~250 GB | ~256 GB | Yes |
| Llama 3.1 8B | 16 GB | ~670 GB | ~686 GB | Yes |
| Llama 3.1 8B @ 8192 seq | 16 GB | ~1.3 TB | ~1.3 TB | Yes |

An 8B model with 4096 context and 1024 population members fits comfortably. That model can actually do QASPER.

**2. Population diversity: memory limits forced ~64 threads on the 3090 Ti. Now we can run 1,000+.**

| Seq length | Per-thread KV (8B model) | Max population in 1.5 TB (500 GB headroom) |
|------------|--------------------------|---------------------------------------------|
| 2048 | ~335 MB | ~4,400 |
| 4096 | ~670 MB | ~2,200 |
| 8192 | ~1.34 GB | ~1,100 |

Population sizes of 1,024-2,048 at QASPER-length contexts are feasible. This is the regime where EGGROLL's rank-1 perturbations become effectively full-rank (N > min(a,b)).

**3. Iteration speed: each epoch takes seconds, not minutes.**

v6e has 918 bf16 TFLOPS per chip. A 64-chip pod delivers ~59,000 TFLOPS. After one-time compilation, hyperparameter sweeps (sigma, learning rate, alternating schedules) can be completed in hours instead of days.

### Revised strategy

Given TPU access, the toy-task detour is unnecessary. Go directly to the real experiment.

| Property | Before (3090 Ti) | Now (v6e-64) |
|----------|-------------------|--------------|
| Model | TinyLlama 1.1B, bucket=256 | Llama 3.1 8B, bucket=4096 |
| Population | 64-256 | 1024-2048 |
| Task | fastzero first, QASPER later | QASPER directly |
| Epoch time | ~minutes | ~seconds |
| Time to Phase 4 | Months | Weeks |

### TPU allocation strategy

- **v6e-64 spot** (europe-west4-a or us-east1-d): Primary training runs. Best compute (918 TFLOPS/chip), 2 TB HBM, 3.3x faster than v4.
- **v4-32 on-demand** (us-central2-b): Debugging and compilation. Will not be preempted mid-debug session. 1 TB HBM is still 40x the 3090 Ti.
- **v5e-64 spot**: Secondary training or parallel sweeps. Lower HBM per chip (16 GB) means smaller populations, but 64 chips still provide 1 TB total. Cost-efficient for ablations.

### Spot preemption and checkpointing

Spot TPU VMs receive a 30-second warning before preemption. No guaranteed minimum runtime. Mitigation:

- Checkpoint `noiser_params` + `params` every N epochs (N=10-50 depending on epoch speed).
- Use `orbax` or simple `jnp.save` to a GCS bucket for fast writes.
- Wrap training loop in a restart-aware script that resumes from the latest checkpoint.
- Run compilation and debugging on the on-demand v4-32 to avoid losing compile time to preemption.

### Revised implementation plan (30-day timeline)

**Week 1: Infrastructure**
- Add Llama 3.1 8B to the model registry (same architecture as TinyLlama, just bigger — a few hours of work).
- Implement QASPER task in `llm_bandits.py` with token-F1 fitness scoring.
- Set up TPU environment: JAX + TPU runtime, GCS bucket for checkpoints, spot preemption handling.
- Verify TinyLlama tests pass on TPU. Then verify 8B model loads and forward-passes on v4-32 on-demand.

**Week 2: Baselines**
- Run EGGROLL-only on QASPER with 8B (no eviction). Establish that the model produces nonzero F1 and that EGGROLL improves it.
- Implement a simple learned eviction policy (small MLP or attention-sum scorer, ~100-500 params, CMA-ES trained). Do not port full NAMM yet.
- Run eviction-only baseline on QASPER with frozen 8B model. Establish NAMM-equivalent baseline.

**Week 3: Core experiment**
- Implement alternating EGGROLL + CMA-ES co-training loop.
- Run joint training on QASPER. Compare against both baselines.
- If positive: run ablations (LoRA rank, sigma, alternating schedule, eviction rate).

**Week 4: Analysis and write-up**
- Analyze attention pattern changes pre/post co-training.
- Run additional ablations if needed.
- If results support the hypothesis: begin full NAMM port (BAM + STFT) as the eviction policy upgrade.
- Write up results.

### The binding constraint

The 30-day TPU clock is now the primary constraint — not memory, not compute, not model size. Every day spent on infrastructure is a day not spent on the core experiment. Prioritize getting the co-training loop running on TPU within the first week.

---

## References

- Cetin et al. "An Evolved Universal Transformer Memory." arXiv:2410.13166 (2024). [Paper](https://arxiv.org/abs/2410.13166) | [Code](https://github.com/SakanaAI/evo-memory)
- Sarkar et al. "Evolution Strategies at the Hyperscale." arXiv:2511.16652 (2025). [Paper](https://arxiv.org/abs/2511.16652) | [Code](https://github.com/ESHyperscale/HyperscaleES)
- Dasigi et al. "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." NAACL 2021. [Paper](https://arxiv.org/abs/2105.03011) | [Dataset](https://huggingface.co/datasets/allenai/qasper)
- Cai et al. "MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection." ICLR 2025. [Paper](https://openreview.net/pdf?id=BQwsRy1h3U)
- Zhang et al. "Attention-Gate: In-context KV-Cache Eviction for Efficient LLMs." arXiv:2410.12876 (2024). [Paper](https://arxiv.org/abs/2410.12876)
- Peng et al. "EvolKV: Evolutionary KV Cache Compression for LLM Inference." EMNLP 2025 Findings. [Paper](https://aclanthology.org/2025.findings-emnlp.88/)
- Liu et al. "Learning to Evict from Key-Value Cache." arXiv:2602.10238 (2026). [Paper](https://arxiv.org/abs/2602.10238)
- Sarkar et al. "ESSA: Evolutionary Strategies for Scalable Alignment." arXiv:2507.04453 (2025). [Paper](https://arxiv.org/abs/2507.04453)
