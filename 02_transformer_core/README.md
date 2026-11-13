# Milestone 02: Modern Transformer Core Architecture

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of this milestone.

## 📄 Primary Papers for this Milestone

- [`06_2017_vaswani_attention_is_all_you_need.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/02_Transformer/06_2017_vaswani_attention_is_all_you_need.pdf)
- [`08_2019_radford_gpt2_unsupervised_multitask_learners.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/02_Transformer/08_2019_radford_gpt2_unsupervised_multitask_learners.pdf)
- [`11_2019_zhang_rmsnorm.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/02_Transformer/11_2019_zhang_rmsnorm.pdf)
- [`12_2020_shazeer_swiglu_glu_variants.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/02_Transformer/12_2020_shazeer_swiglu_glu_variants.pdf)
- [`13_2023_ainslie_gqa_grouped_query_attention.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/02_Transformer/13_2023_ainslie_gqa_grouped_query_attention.pdf)
- [`16_2021_su_roformer_rotary_position_embedding_rope.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/04_Positional_Encodings/16_2021_su_roformer_rotary_position_embedding_rope.pdf)

---

### Mathematical Formulation
1. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
2. **RoPE Complex Rotation**:
   $$R_{\Theta, m}^d = \text{diag}(R_{\theta_1, m}, \dots, R_{\theta_{d/2}, m})$$
3. **SwiGLU Gating**:
   $$\text{SwiGLU}(x) = (x W_1 \odot \text{silu}(x W_3)) W_2$$

---

## 🛠️ Code Artifacts to Build

| Script Filename | Implementation Focus | Status |
|---|---|:---:|
| `scaled_dot_product_attention.py` | Multi-Head Attention and Grouped-Query Attention (GQA) with causal lower-triangular masking | ⚪ Pending |
| `rotary_position_embedding.py` | RoPE 2D complex plane rotation matrices applied to Query and Key projections | ⚪ Pending |
| `rmsnorm.py` | Root Mean Square Layer Normalization eliminating mean centering | ⚪ Pending |
| `swiglu_ffn.py` | 3-matrix Swish-Gated Linear Unit feed-forward sublayer | ⚪ Pending |
| `modern_transformer_decoder.py` | Complete consolidated modern foundation model (LLaMA/Mistral style) | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper to Reproduce**: `Paper 06: Vaswani et al. (2017) — "Attention Is All You Need"`

- **Next Action**: Create `scaled_dot_product_attention.py` and verify with analytical gradients.
