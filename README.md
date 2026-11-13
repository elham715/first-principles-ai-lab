# First-Principles AI Lab 🧬⚡
### *Deconstructing & Reproducing Modern LLM Intelligence from Raw Tensor Mathematics*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In%20Active%20Research-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> *"If you really learn all of these, you'll know 90% of what matters today."*  
> — **Ilya Sutskever** (to John Carmack on mastering AI from first principles)

---

## 🎯 Mission & Philosophy

Most software engineers in the modern AI era are **"API Wrappers"** or **"Library Callers"**. They import high-level packages (`from transformers import ...`), call `.from_pretrained()`, and treat intelligence as an opaque black box.

This repository is dedicated to the **Top 1% Engineering Philosophy**:
1. **Zero External Black Boxes**: No `nn.MultiheadAttention`, no Hugging Face high-level wrappers, no opaque layers.
2. **Raw Tensor Math First**: Every mechanism is derived on paper, written in raw PyTorch/NumPy tensor operations, and verified against analytical derivatives.
3. **The Lineage of Intelligence**: We trace how sequence processing evolved from **Elman Recurrence (1990)** to **LSTMs (1997)**, **Additive Attention (2014)**, **Pure Transformers (2017)**, and finally to modern production foundation models (**LLaMA, SwiGLU, RoPE, GQA, LoRA, and DPO**).

---

## 🗺️ The "Titan 6" Master Roadmap

Rather than building 58 separate fragmented scripts, our multi-month roadmap consolidates the seminal literature into **6 Production-Grade Milestone Engines**:

```
                                  THE EVOLUTIONARY LINEAGE
                                  
   [ 1990 Elman RNN ] ──► [ 1997 LSTM ] ──► [ 2014 Bahdanau Attention ]
                                                       │
                                                       ▼
   [ 2023 Modern LLaMA ] ◄── [ 2019 GPT-2 Pre-LN ] ◄── [ 2017 Vaswani Transformer ]
   (RoPE + SwiGLU + GQA)
           │
           ├───────────────────────────────┬───────────────────────────────┐
           ▼                               ▼                               ▼
   [ 04. PEFT & LoRA ]           [ 05. Alignment / DPO ]       [ 06. Fast Inference ]
   (Rank-r Decomposition)        (Closed-Form Loss)            (KV-Cache + Speculative)
```

| Milestone Engine | Target Directory | Seminal Papers Unified in Code | Core Theoretical Breakthrough | Status |
|---|---|---|---|:---:|
| **00. Coding Foundations** | [`00_coding_foundations/`](./00_coding_foundations/) | 80/20 Prerequisite Notebooks | Python OOP, NumPy, Pandas, PyTorch & TensorFlow core mechanics | 🟢 Ready |
| **01. Sequence Foundations** | [`01_sequence_foundations/`](./01_sequence_foundations/) | Elman (1990), Hochreiter (1997), Bengio (2003), Sutskever (2014), Bahdanau (2014) | BPTT analytical gradients (`≤ 6.93e-11`), Constant Error Carousel (CEC), Additive Soft Attention | 🟡 In Progress |
| **02. Modern Transformer Core** | [`02_transformer_core/`](./02_transformer_core/) | Vaswani (2017), Radford (2019), Zhang (2019), Shazeer (2020), Su (2021), Ainslie (2023) | Full Causal LLM: Scaled Dot-Product Attention, RoPE 2D rotations, RMSNorm, SwiGLU, GQA | ⚪ Planned |
| **03. Residual Highway & Stability** | [`03_residual_stream/`](./03_residual_stream/) | He (2015, 2016), Veit (2016), Xiong (2020), Wang (DeepNet 2022) | Identity shortcuts $x + F(x)$, shallow ensemble proof ($2^L$ paths), Pre-LN vs Post-LN variance proofs | ⚪ Planned |
| **04. Parameter-Efficient Fine-Tuning** | [`04_peft_lora/`](./04_peft_lora/) | Hu (2021), Zhang (2023), Li (2022, 2024), Pfeiffer (2021) | Low-Rank Adaptation $\Delta W = BA$, zero-init attention gating, Mixture-of-LoRA-Experts | ⚪ Planned |
| **05. Alignment & Preference Optimization** | [`05_alignment_dpo/`](./05_alignment_dpo/) | Rafailov (2023), Zhou (LIMA 2023), Wei (CoT 2022), Hoffmann (2022) | Direct Preference Optimization (DPO) closed-form binary cross-entropy, Supervised Alignment | ⚪ Planned |
| **06. High-Throughput Inference & Systems** | [`06_inference_systems/`](./06_inference_systems/) | Dao (FlashAttention 2022), Leviathan (2022), Sheng (S-LoRA 2023) | Dynamic Paged KV-Cache, speculative drafting & parallel token verification | ⚪ Planned |

---

## 🔬 Directory Architecture

```
first-principles-ai-lab/
├── README.md                           # Master Architecture & Milestone Roadmap
├── 00_coding_foundations/              # PREREQUISITE: 80/20 Python OOP, NumPy, Pandas, PyTorch & TF
│   ├── 00a_python_and_oop_80_20.ipynb
│   ├── 00b_numpy_80_20.ipynb
│   ├── 00c_pandas_80_20.ipynb
│   ├── 00d_pytorch_core_80_20.ipynb
│   └── 00e_tensorflow_core_80_20.ipynb
├── 01_sequence_foundations/            # MILESTONE 1: Recurrence to Attention
│   ├── 01_elman_rnn_numpy.py           # Elman cell with manual analytical BPTT (verified)
│   ├── 01_elman_rnn_walkthrough.ipynb  # Interactive GradCheck & Vanishing Gradient curves
│   ├── 02_lstm_cell_from_scratch.py    # Gated LSTM with Constant Error Carousel (planned)
│   ├── 03_seq2seq_encoder_decoder.py   # Sutskever Seq2Seq with reversed sequences (planned)
│   └── 04_bahdanau_attention.py        # Additive Attention with dynamic alignment heatmaps (planned)
├── 02_transformer_core/                # MILESTONE 2: The Modern Foundation Backbone
│   ├── scaled_dot_product_attention.py # Vectorized multi-head attention
│   ├── rotary_position_embedding.py    # RoPE 2D complex rotation matrices
│   ├── rmsnorm.py                      # Root Mean Square normalization
│   ├── swiglu_ffn.py                   # Gated Linear Unit Feed-Forward
│   └── modern_transformer_decoder.py   # Consolidated modern foundation LLM (LLaMA-style)
├── 03_residual_stream/                 # MILESTONE 3: Gradient Dynamics & Highway Stability
│   ├── pre_vs_post_ln_analysis.py      # Numerical variance & gradient norm tracker across layers
│   └── ensemble_path_analysis.py       # Veit ensemble sampling ($2^L$ paths)
├── 04_peft_lora/                       # MILESTONE 4: Low-Rank Adaptation & Adapters
│   ├── lora_linear.py                  # Linear layer with low-rank bypass Delta W = B*A
│   └── zero_init_adapter.py            # Gated attention grafting
├── 05_alignment_dpo/                   # MILESTONE 5: Implicit Reward Alignment
│   └── dpo_trainer.py                  # Closed-form DPO loss without PPO
├── 06_inference_systems/               # MILESTONE 6: High-Speed Inference
│   ├── paged_kv_cache.py               # Memory-efficient token caching
│   └── speculative_decoder.py          # Draft model proposal + Target model verification
├── tests/                              # Unit tests comparing manual math to PyTorch autograd
└── benchmarks/                         # Speed, latency, and memory benchmarks
```

---

## 📐 Mathematical Integrity Standards

Every script in this repository must satisfy the **Three Laws of First-Principles Verification**:

1. **Analytical Gradient Matching**:
   Whenever a backward pass is implemented manually, it must match PyTorch's automatic differentiation (`torch.autograd.gradcheck`) within numerical tolerance:
   $$\max \left| \nabla_{\text{analytical}} - \nabla_{\text{autograd}} \right| < 10^{-5}$$

2. **Dimension Hygiene**:
   Every tensor operation must feature explicit comment annotations detailing its batch and sequence shapes:
   ```python
   # q: [batch, n_heads, seq_len, head_dim]
   # k: [batch, n_heads, seq_len, head_dim]
   # scores: [batch, n_heads, seq_len, seq_len]
   ```

3. **Zero Duration Hallucination**:
   All training and inference benchmarks report exact wall-clock times, throughput tokens-per-second, and peak VRAM allocated.

---

## 📚 Reading Curriculum & Lineage

The theoretical foundations implemented here are drawn directly from **58 Seminal Papers** spanning Recurrence, Attention, Transformers, Residual Connections, PEFT/LoRA, Alignment/DPO, and Inference Systems (including Ilya Sutskever's 30 Papers reading list). Each milestone directory links directly to the mathematical formulations and verification code for its corresponding papers.

---

## 🚀 Getting Started

```bash
# Clone and enter the laboratory
cd first-principles-ai-lab

# Verify environment
python3 -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA/MPS Available:', torch.cuda.is_available() or torch.backends.mps.is_available())"

# Run the first reproduction milestone tests
pytest tests/ -v
```

---

### Authorship & Journey
* **Laboratory Architect**: Elham's Universe AI Research
* **Timeline**: Multi-Month Rigorous Masterclass (November 2026 – 2027)
* *"Code is the ultimate test of understanding."*

