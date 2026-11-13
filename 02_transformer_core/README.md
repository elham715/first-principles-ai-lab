# 02. Transformer Core Architecture

From-scratch components for modern autoregressive Transformer decoders.

## Papers

1. Vaswani et al. (2017) — *Attention Is All You Need*
2. Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2)
3. Zhang & Sennrich (2019) — *Root Mean Square Layer Normalization*
4. Shazeer (2020) — *GLU Variants Improve Transformer* (SwiGLU)
5. Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
6. Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models*

## Core Equations

* **Scaled Dot-Product Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
* **SwiGLU:**
  $$\text{SwiGLU}(x) = (x W_1 \odot \text{silu}(x W_3)) W_2$$

## Planned Files

| File | Description | Status |
|---|---|---|
| `scaled_dot_product_attention.py` | Multi-head and grouped-query attention with causal masking | Planned |
| `rotary_position_embedding.py` | 2D complex rotation embeddings (RoPE) on queries and keys | Planned |
| `rmsnorm.py` | Root mean square layer normalization | Planned |
| `swiglu_ffn.py` | Swish-gated feed-forward block | Planned |
| `modern_transformer_decoder.py` | Full causal decoder combining the components above | Planned |
