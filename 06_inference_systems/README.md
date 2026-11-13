# 06. Inference Systems & Speculative Decoding

Memory-efficient KV caching and draft-target speculative verification.

## Papers

1. Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
2. Leviathan et al. (2022) — *Fast Inference from Transformers via Speculative Decoding*

## Core Acceptance Rule

$$\text{Accept } x_{n+i} \text{ with probability } \min\left(1, \frac{p_{\text{target}}(x_{n+i}|x_{<n+i})}{q_{\text{draft}}(x_{n+i}|x_{<n+i})}\right)$$

## Planned Files

| File | Description | Status |
|---|---|---|
| `paged_kv_cache.py` | Block-based KV cache avoiding repeated tensor reallocation | Planned |
| `speculative_decoder.py` | Draft generation and parallel target verification | Planned |
