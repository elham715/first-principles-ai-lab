# 04. Parameter-Efficient Fine-Tuning (LoRA & Adapters)

Low-rank weight updates and zero-initialized adapter grafting without modifying frozen base weights.

## Papers

1. Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
2. Zhang et al. (2023) — *LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention*
3. Li et al. (2024) — *MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts*

## Core Equation

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r},\; A \in \mathbb{R}^{r \times d_{\text{in}}}$$

Initializing $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$ keeps $\Delta W = 0$ at step 0.

## Planned Files

| File | Description | Status |
|---|---|---|
| `lora_linear.py` | Linear wrapper with frozen $W_0$ and trainable rank-$r$ factors $B, A$ | Planned |
| `zero_init_adapter.py` | Zero-initialized gated attention adapter | Planned |
