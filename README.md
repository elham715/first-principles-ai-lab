# first-principles-ai-lab

From-scratch implementations of foundational sequence, attention, transformer, and alignment papers in NumPy and PyTorch.

The goal is simple: build each architecture directly from its mathematical formulation, verify analytical gradients against finite differences or `torch.autograd`, and document where each design breaks without relying on high-level wrapper APIs.

---

## Structure

```text
first-principles-ai-lab/
├── 00_coding_foundations/      # 80/20 notebooks: Python OOP, NumPy, Pandas, PyTorch, TensorFlow
├── 01_sequence_foundations/    # Elman RNN (1990), LSTM (1997), Seq2Seq (2014), Bahdanau (2014)
├── 02_transformer_core/        # Attention, RoPE, RMSNorm, SwiGLU, GQA, causal decoder
├── 03_residual_stream/         # ResNet identity mappings, Pre-LN vs Post-LN stability
├── 04_peft_lora/               # LoRA (rank-r adapters), zero-init attention grafting
├── 05_alignment_dpo/           # Direct Preference Optimization (DPO)
├── 06_inference_systems/       # KV-cache paging and speculative decoding
├── tests/                      # Unit tests and gradient checks
└── benchmarks/                 # Timing and memory benchmarks
```

---

## Roadmap

| Directory | Focus | Papers Covered | Status |
|---|---|---|---|
| [`00_coding_foundations/`](./00_coding_foundations/) | Prerequisites | Python OOP, NumPy, Pandas, PyTorch, TensorFlow | Done |
| [`01_sequence_foundations/`](./01_sequence_foundations/) | Recurrence & Attention | Elman (1990), Hochreiter & Schmidhuber (1997), Sutskever et al. (2014), Bahdanau et al. (2014) | Done |
| [`02_transformer_core/`](./02_transformer_core/) | Transformer Decoder | Vaswani et al. (2017), Radford et al. (2019), Zhang & Sennrich (2019), Shazeer (2020), Su et al. (2021), Ainslie et al. (2023) | Planned |
| [`03_residual_stream/`](./03_residual_stream/) | Deep Optimization | He et al. (2015, 2016), Veit et al. (2016), Xiong et al. (2020), Wang et al. (2022) | Planned |
| [`04_peft_lora/`](./04_peft_lora/) | Parameter-Efficient Tuning | Hu et al. (2021), Zhang et al. (2023), Pfeiffer et al. (2021), Li et al. (2024) | Planned |
| [`05_alignment_dpo/`](./05_alignment_dpo/) | Alignment | Rafailov et al. (2023), Zhou et al. (2023), Wei et al. (2022), Hoffmann et al. (2022) | Planned |
| [`06_inference_systems/`](./06_inference_systems/) | Inference Systems | Dao et al. (2022), Leviathan et al. (2022), Sheng et al. (2023) | Planned |

---

## Verification Checks

1. **Gradient checking:** Manual backward passes are checked against finite differences or PyTorch autograd (`relative error < 1e-5`).
2. **Shape annotations:** Tensor dimensions (`(T, B, D)`, `(B, H, T, d_k)`, etc.) are noted explicitly alongside matrix operations.

---

## Running Locally

```bash
git clone https://github.com/elham715/first-principles-ai-lab.git
cd first-principles-ai-lab

python3 01_sequence_foundations/01_elman_rnn_numpy.py
```

Requires Python 3.10+, `numpy`, `pandas`, `torch`, and `matplotlib`.
