# Milestone 06: High-Speed Inference Systems & Speculative Decoding

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of this milestone.

## 📄 Primary Papers for this Milestone

- [`37_2022_dao_flashattention.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/08_Inference_and_Optimization/37_2022_dao_flashattention.pdf)
- [`38_2022_leviathan_speculative_decoding.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/08_Inference_and_Optimization/38_2022_leviathan_speculative_decoding.pdf)

---

### Mathematical Formulation
1. **Speculative Acceptance Criterion**:
   $$\text{Accept token } x_{n+i} \text{ with probability } \min\left(1, \frac{p_{\text{target}}(x_{n+i}|x_{<n+i})}{q_{\text{draft}}(x_{n+i}|x_{<n+i})}\right)$$

---

## 🛠️ Code Artifacts to Build

| Script Filename | Implementation Focus | Status |
|---|---|:---:|
| `paged_kv_cache.py` | Simulated dynamic block-paging KV-cache avoiding tensor reallocations | ⚪ Pending |
| `speculative_decoder.py` | Parallel draft proposal generation and target model acceptance verification | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper to Reproduce**: `Paper 38: Yaniv Leviathan et al. (2022) — "Fast Inference from Transformers via Speculative Decoding"`

- **Next Action**: Create `paged_kv_cache.py` and verify with analytical gradients.
